"""CPU policy, training-mask parity and reference-logit tests; no server needed."""

import importlib.util
import json
import math
import os
import random
import sys
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load complete modules without importing the GPU serving package.
register_cpu_ci = load_module(
    "_reasoning_ci", ROOT / "python/sglang/test/ci/ci_register.py"
).register_cpu_ci
register_cpu_ci(est_time=10, suite="base-a-test-cpu")

eviction = load_module(
    "_reasoning_eviction", ROOT / "python/sglang/srt/mem_cache/reasoning_eviction.py"
)
Policy = eviction.ReasoningEvictionPolicy
Span = eviction.ReasoningSpan
Message = eviction.MessageBoundary
Markers = eviction.IFMReasoningMarkers
Reference = eviction.ReasoningEvictionReference


class Tokenizer:
    """One deterministic vocabulary implementing both HF and xLLM interfaces."""

    bos_id, eos_id, unk_token_id = 0, 1, 511

    def __init__(self):
        names = [
            "<|ifm|im_start|>",
            "<|ifm|im_end|>",
            "system",
            "user",
            "assistant",
            "tool",
            "\n",
        ]
        names += [
            f"<{slash}ifm|{kind}>"
            for kind in ("think", "think_fast", "think_faster")
            for slash in ("", "/")
        ]
        self.ids = {name: 100 + i for i, name in enumerate(names)}

    def encode(self, text, **kwargs):
        return [self.ids[text]]


def fixture(lengths=(12, 9, 3), *, final_completed=True):
    """User -> assistant -> tool -> assistant -> user -> assistant -> tool."""
    tokenizer = Tokenizer()
    ids = tokenizer.ids
    tokens, messages, masks, conversation = [0], [], [0], []

    def add(role, body, kind=None, completed=True):
        start = len(tokens)
        header = [ids["<|ifm|im_start|>"], ids[role], ids["\n"]]
        message = header + body + ([ids["<|ifm|im_end|>"]] if completed else [])
        tokens.extend(message)
        masks.extend(
            [0] * len(header)
            + [int(role in ("assistant", "tool"))] * (len(message) - len(header))
        )
        messages.append(Message(role, start, start + 2, len(tokens), completed))
        source = {"role": role, "content": "body"}
        if kind:
            source[kind] = "reasoning"
        conversation.append(source)

    quoted = [ids["<ifm|think>"], 20, ids["</ifm|think>"]]
    add("user", quoted)
    for turn, (kind, length) in enumerate(
        zip(("think", "think_fast", "think_faster"), lengths)
    ):
        completed = final_completed or turn != 2
        add(
            "assistant",
            [
                ids[f"<ifm|{kind}>"],
                *[10 + j % 60 for j in range(length)],
                ids[f"</ifm|{kind}>"],
                80,
                81,
            ],
            kind,
            completed,
        )
        if completed:
            add("tool" if turn != 1 else "user", quoted)
    if final_completed:
        tokens.append(1)
        masks.append(0)
    return tokenizer, tokens, messages, masks, conversation


def reference(window=1, **kwargs):
    tokenizer, tokens, messages, _, _ = fixture(**kwargs)
    spans = eviction.annotate_reasoning(
        tokens, messages, Markers.from_tokenizer(tokenizer)
    )
    return Reference(tokens, spans, Policy(window))


class PolicyTests(unittest.TestCase):
    def test_window_semantics_and_protected_delimiters(self):
        span = Span(3, 15, 20)
        for window, expected in (
            (None, []),
            (0, list(range(3, 15))),
            (1, list(range(3, 15))),
            (8, list(range(3, 8))),
            (99, []),
        ):
            with self.subTest(window=window):
                self.assertEqual(list(Policy(window).dropped_positions(span)), expected)
                self.assertNotIn(span.start - 1, expected)
                self.assertNotIn(span.stop, expected)

    def test_invalid_policy_and_fixed_policy(self):
        for window in (-1, True, False, 1.5, "8"):
            with self.subTest(window=window), self.assertRaises(ValueError):
                Policy(window)
        for kwargs in ({"boundary": "next_user"}, {"version": 2}, {"version": True}):
            with self.assertRaises(ValueError):
                Policy(**kwargs)
        with self.assertRaises(FrozenInstanceError):
            Policy(8).window = 1

    def test_span_validation(self):
        for args in ((0, 2, 4), (4, 3, 6), (1, 3, 3), (-1, 2, 4), (True, 2, 4)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Span(*args)
        self.assertEqual(list(Policy(0).dropped_positions(Span(1, 1, 3))), [])

    def test_invalid_trace_spans(self):
        for spans in (
            (Span(1, 5, 8), Span(4, 6, 8)),
            (Span(1, 2, 8), Span(3, 5, 8)),  # shared opening/closing marker
            (Span(4, 5, 8), Span(1, 2, 8)),
            (Span(1, 2, 8), Span(5, 6, 10)),
            (Span(1, 2, 12),),
        ):
            with self.subTest(spans=spans), self.assertRaises(ValueError):
                Reference(range(10), spans, Policy(1))

    def test_invalid_token_ids(self):
        for token in (-1, True, 2.5, "42", [42]):
            with self.subTest(token=token), self.assertRaises(ValueError):
                Reference([token], [], Policy(1))

    def test_multi_span_window_applies_separately(self):
        spans = (Span(2, 12, 28), Span(15, 25, 28))
        ref = Reference(range(30), spans, Policy(8))
        dropped = {i for i, expiry in enumerate(ref.expires_at) if expiry is not None}
        self.assertEqual(dropped, {2, 3, 4, 15, 16, 17})

    def test_terminal_token_sees_reasoning_then_tool_loses_it(self):
        ref = reference()
        for span in ref.spans:
            self.assertIn(span.start, ref.visible_positions(span.expires_at - 1))
            self.assertNotIn(span.start, ref.visible_positions(span.expires_at))
            self.assertIn(span.start - 1, ref.visible_positions(span.expires_at))
            self.assertIn(span.stop, ref.visible_positions(span.expires_at))

    def test_cumulative_tool_cycles_keep_all_current_reasoning(self):
        ref = reference()
        for turn, current in enumerate(ref.spans):
            visible = ref.visible_positions(current.expires_at - 1)
            self.assertTrue(set(range(current.start, current.stop)).issubset(visible))
            for old in ref.spans[:turn]:
                self.assertTrue(set(range(old.start, old.stop)).isdisjoint(visible))

    def test_all_non_reasoning_tokens_are_protected(self):
        ref = reference(0)
        reasoning = {p for span in ref.spans for p in range(span.start, span.stop)}
        layout = ref.layout_after(len(ref.token_ids))
        self.assertEqual(
            set(layout.original_positions), set(range(len(ref.token_ids))) - reasoning
        )


class AnnotationTests(unittest.TestCase):
    def test_all_horizon_modes_and_prompt_priming(self):
        tokenizer, tokens, messages, masks, _ = fixture()
        spans = eviction.annotate_reasoning(
            tokens, messages, Markers.from_tokenizer(tokenizer)
        )
        self.assertEqual([s.stop - s.start for s in spans], [12, 9, 3])
        self.assertEqual(
            [s.expires_at for s in spans],
            [m.end for m in messages if m.role == "assistant"],
        )
        # Eligibility is not limited to the sampled suffix: opening + initial
        # reasoning can have been supplied in the prompt with a zero loss mask.
        masks[spans[0].start - 1 : spans[0].start + 4] = [0] * 5
        self.assertEqual(
            eviction.annotate_reasoning(
                tokens, messages, Markers.from_tokenizer(tokenizer)
            ),
            spans,
        )

    def test_quoted_tags_in_user_tool_and_answer_are_ignored(self):
        tokenizer, tokens, messages, _, _ = fixture()
        assistant = messages[1]
        # Existing think tags now occur after ordinary answer text.
        tokens[assistant.body_start] = 90  # replace the optional newline
        spans = eviction.annotate_reasoning(
            tokens, messages, Markers.from_tokenizer(tokenizer)
        )
        self.assertEqual(len(spans), 2)
        self.assertTrue(all(s.start > assistant.end for s in spans))

    def test_empty_think_blocks(self):
        ref = reference(0, lengths=(0, 0, 0))
        self.assertEqual(len(ref.spans), 3)
        self.assertTrue(all(s.start == s.stop for s in ref.spans))
        self.assertEqual(ref.layout_after(len(ref.token_ids)).token_ids, ref.token_ids)

    def test_unfinished_or_cancelled_message_is_not_a_boundary(self):
        ref = reference(0, final_completed=False)
        self.assertEqual(len(ref.spans), 2)
        self.assertIn(
            len(ref.token_ids) - 3,
            ref.layout_after(len(ref.token_ids)).original_positions,
        )
        tokenizer, tokens, messages, _, _ = fixture(final_completed=False)
        # An unfinished think block must also stay intact.
        tokens = tokens[:-3]
        messages[-1] = replace(messages[-1], end=len(tokens))
        self.assertEqual(
            len(
                eviction.annotate_reasoning(
                    tokens, messages, Markers.from_tokenizer(tokenizer)
                )
            ),
            2,
        )

    def test_malformed_reasoning_is_retained(self):
        tokenizer, tokens, messages, _, _ = fixture()
        markers = Markers.from_tokenizer(tokenizer)
        first = eviction.annotate_reasoning(tokens, messages, markers)[0]
        for position, replacement in (
            (first.stop, 42),  # missing close
            (first.start + 1, markers.think_pairs[1][0]),  # nested open
            (first.start + 1, markers.think_pairs[1][1]),  # mismatched close
            (first.stop + 1, markers.think_pairs[0][1]),  # duplicate close
        ):
            candidate = list(tokens)
            candidate[position] = replacement
            with self.subTest(position=position, replacement=replacement):
                spans = eviction.annotate_reasoning(candidate, messages, markers)
                self.assertEqual(len(spans), 2)

    def test_message_metadata_validation(self):
        tokenizer, tokens, messages, _, _ = fixture()
        markers = Markers.from_tokenizer(tokenizer)
        candidates = [
            [messages[1], messages[0]],
            [replace(messages[0], end=len(tokens) + 1)],
            [replace(messages[0], end=messages[0].end - 1)],
            [replace(messages[0], start=0)],
            [replace(messages[0], completed=False), *messages[1:]],
        ]
        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                eviction.annotate_reasoning(tokens, candidate, markers)
        candidate = list(tokens)
        candidate[messages[2].body_start + 2] = markers.message_start
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            eviction.annotate_reasoning(candidate, messages, markers)

    def test_marker_resolution_rejects_unknown_split_and_colliding_tokens(self):
        for result in ([500, 501], [511], [100]):
            tokenizer = SimpleNamespace(
                encode=lambda text, **kwargs: result, unk_token_id=511
            )
            with self.subTest(result=result), self.assertRaises(ValueError):
                Markers.from_tokenizer(tokenizer)

    def test_multi_token_newline(self):
        markers = Markers(100, 101, ((107, 108),), (106, 109))
        tokens = [100, 104, 106, 109, 107, 42, 108, 101]
        self.assertEqual(
            eviction.annotate_reasoning(
                tokens, [Message("assistant", 0, 2, 8)], markers
            ),
            (Span(5, 6, 8),),
        )


class LayoutAndReplayTests(unittest.TestCase):
    def test_absolute_positions_survive_repeated_compaction(self):
        ref = reference()
        previous_drops = 0
        for span in ref.spans:
            layout = ref.layout_after(span.expires_at)
            previous_drops += span.stop - span.start
            self.assertEqual(layout.next_absolute_position, span.expires_at)
            self.assertEqual(len(layout.token_ids), span.expires_at - previous_drops)
            self.assertEqual(
                layout.token_ids,
                tuple(ref.token_ids[p] for p in layout.original_positions),
            )
            self.assertEqual(layout.original_positions[-1], span.expires_at - 1)

    def test_compaction_requires_terminal_forward_and_correct_layout(self):
        ref = reference()
        boundary = ref.spans[1].expires_at
        before = ref.visible_positions(boundary - 1)
        for committed in (boundary - 1, boundary + 1, len(before)):
            with self.subTest(committed=committed), self.assertRaises(ValueError):
                ref.plan_compaction(boundary, before, committed_through=committed)
        with self.assertRaises(ValueError):
            ref.plan_compaction(boundary, range(boundary), committed_through=boundary)
        with self.assertRaises(ValueError):
            ref.plan_compaction(boundary - 1, before, committed_through=boundary - 1)

    def test_gather_matches_layout_and_is_repeatable(self):
        for window in (None, 0, 1, 8, 999):
            ref = reference(window)
            for span in ref.spans:
                before = ref.visible_positions(span.expires_at - 1)
                plan = ref.plan_compaction(
                    span.expires_at, before, committed_through=span.expires_at
                )
                self.assertEqual(
                    tuple(before[i] for i in plan.keep_indices),
                    plan.layout.original_positions,
                )
                self.assertEqual(
                    set(plan.keep_indices) | set(plan.drop_indices),
                    set(range(len(before))),
                )
                self.assertTrue(set(plan.keep_indices).isdisjoint(plan.drop_indices))
                self.assertEqual(
                    plan,
                    ref.plan_compaction(
                        span.expires_at, before, committed_through=span.expires_at
                    ),
                )

    def test_final_boundary_prunes_only_after_terminal_forward(self):
        original = reference(0)
        boundary = original.spans[-1].expires_at
        ref = Reference(original.token_ids[:boundary], original.spans, original.policy)
        span = ref.spans[-1]
        self.assertIn(span.start, ref.visible_positions(boundary - 1))
        self.assertNotIn(span.start, ref.layout_after(boundary).original_positions)
        self.assertEqual(
            ref.replay_steps()[-1].drop_after, tuple(range(span.start, span.stop))
        )

    def test_chunked_replay_preserves_every_boundary_and_layout(self):
        for window in (None, 0, 1, 8, 999):
            ref = reference(window)
            for chunk in (None, 1, 7, 16, 4096):
                resident, processed = [], 0
                for step in ref.replay_steps(chunk):
                    self.assertEqual(step.start, processed)
                    self.assertFalse(
                        any(step.start < s.expires_at < step.stop for s in ref.spans)
                    )
                    resident.extend(range(step.start, step.stop))
                    resident = [p for p in resident if p not in step.drop_after]
                    self.assertEqual(
                        tuple(resident), ref.layout_after(step.stop).original_positions
                    )
                    processed = step.stop
                self.assertEqual(processed, len(ref.token_ids))

    def test_empty_trace_and_invalid_lengths(self):
        ref = Reference([], [], Policy(8))
        self.assertEqual(ref.replay_steps(), ())
        self.assertEqual(ref.attention_mask(), ())
        self.assertEqual(ref.layout_after(0).original_positions, ())
        for chunk in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                ref.replay_steps(chunk)
        for position in (-1, 1, True):
            with self.assertRaises(ValueError):
                ref.layout_after(position)
        with self.assertRaises(ValueError):
            ref.visible_positions(0)

    def test_frozen_training_fixture(self):
        golden = json.loads(
            (
                Path(__file__).parent / "fixtures/reasoning_eviction_training.json"
            ).read_text()
        )
        tokenizer, tokens, messages, _, _ = fixture()
        self.assertEqual(tokens, golden["token_ids"])
        self.assertEqual([asdict(m) for m in messages], golden["messages"])
        spans = eviction.annotate_reasoning(
            tokens, messages, Markers.from_tokenizer(tokenizer)
        )
        self.assertEqual([asdict(s) for s in spans], golden["spans"])
        for window in (None, 0, 1, 8, 4096):
            ref = Reference(tokens, spans, Policy(window))
            # xLLM uses transcript length for keys that never expire during
            # this transcript; inference keeps None for future continuation.
            actual = [
                len(tokens) if expiry is None else expiry for expiry in ref.expires_at
            ]
            self.assertEqual(actual, golden["expires_at"][str(window)])

    @unittest.skipUnless(
        os.environ.get("XLLM_REASONING_SOURCE"),
        "set XLLM_REASONING_SOURCE for live training-code parity",
    )
    def test_actual_training_annotations_and_every_mask_entry(self):
        import numpy as np

        training = load_module(
            "_xllm_reasoning_parity", os.environ["XLLM_REASONING_SOURCE"]
        )
        rng = random.Random(2718)
        for case in range(30):
            lengths = tuple(rng.randrange(1, 40) for _ in range(3))
            tokenizer, tokens, messages, masks, conversation = fixture(lengths)
            spans = eviction.annotate_reasoning(
                tokens, messages, Markers.from_tokenizer(tokenizer)
            )
            expected = training.annotate_reasoning(
                tokens,
                masks,
                tokenizer,
                boundary="assistant_message",
                conversation=conversation,
            )
            self.assertEqual([(s.start, s.stop, s.expires_at) for s in spans], expected)
            metadata = training.pack_reasoning([(len(tokens), expected)], len(tokens))
            for window in (None, 0, 1, 8, 4096):
                config = SimpleNamespace(
                    window=0 if window is None else window,
                    drop_probability=0.0 if window is None else 1.0,
                    ramp_steps=0,
                )
                expiry = training.make_expiry(metadata, config, 0, 0)["expires_at"]
                positions = np.arange(len(tokens))
                expected_mask = (positions[:, None] >= positions[None, :]) & (
                    positions[:, None] < expiry[None, :]
                )
                ref = Reference(tokens, spans, Policy(window))
                with self.subTest(case=case, window=window):
                    np.testing.assert_array_equal(ref.attention_mask(), expected_mask)


class ReferenceLogitTests(unittest.TestCase):
    """Two-layer CPU transformer checks attention history, not Horizon kernels."""

    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(19)
        cls.embedding = torch.randn(512, 8, generator=generator, dtype=torch.float64)
        cls.weights = [
            torch.randn(4, 8, 8, generator=generator, dtype=torch.float64)
            / math.sqrt(8)
            for _ in range(2)
        ]
        cls.output = torch.randn(8, 32, generator=generator, dtype=torch.float64)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)

    def forward(self, token_ids, positions, cache=None, mask=None):
        torch = self.torch
        x = self.embedding[list(token_ids)]
        positions = torch.tensor(positions)
        angles = positions[:, None] * torch.tensor([1.0, 0.1, 0.01, 0.001])

        def rope(values):
            a, b = values[:, 0::2], values[:, 1::2]
            return torch.stack(
                (
                    a * angles.cos() - b * angles.sin(),
                    a * angles.sin() + b * angles.cos(),
                ),
                dim=-1,
            ).flatten(1)

        new_cache = []
        for layer, weights in enumerate(self.weights):
            q, k, v = rope(x @ weights[0]), rope(x @ weights[1]), x @ weights[2]
            key_positions = positions
            if cache is not None:
                old_k, old_v, old_positions = cache[layer]
                k, v = torch.cat((old_k, k)), torch.cat((old_v, v))
                key_positions = torch.cat((old_positions, positions))
            allowed = (
                positions[:, None] >= key_positions[None, :] if mask is None else mask
            )
            scores = (q @ k.T / math.sqrt(8)).masked_fill(~allowed, -float("inf"))
            x = (x + scores.softmax(-1) @ v @ weights[3]).tanh()
            new_cache.append((k, v, key_positions))
        return x @ self.output, new_cache

    def replay(self, ref, chunk):
        torch = self.torch
        logits, cache = [], None
        for step in ref.replay_steps(chunk):
            output, cache = self.forward(
                ref.token_ids[step.start : step.stop],
                range(step.start, step.stop),
                cache,
            )
            logits.append(output)
            keep = torch.tensor(
                [int(p) not in step.drop_after for p in cache[0][2]], dtype=torch.bool
            )
            cache = [(k[keep], v[keep], positions[keep]) for k, v, positions in cache]
        return torch.cat(logits), cache

    def test_compacted_and_cold_replay_logits_match_dense_training_graph(self):
        torch = self.torch
        for window in (None, 0, 1, 8, 4096):
            ref = reference(window)
            mask = torch.tensor(ref.attention_mask())
            dense, _ = self.forward(ref.token_ids, range(len(ref.token_ids)), mask=mask)
            for chunk in (None, 1, 7, 16):
                with self.subTest(window=window, chunk=chunk):
                    replayed, cache = self.replay(ref, chunk)
                    torch.testing.assert_close(replayed, dense, rtol=1e-10, atol=1e-10)
                    self.assertEqual(
                        tuple(cache[0][2].tolist()),
                        ref.layout_after(len(ref.token_ids)).original_positions,
                    )

    def test_shortened_transcript_reprefill_is_not_faithful_recovery(self):
        ref = reference(1)
        faithful, _ = self.replay(ref, 1)
        layout = ref.layout_after(len(ref.token_ids))
        shortened, _ = self.forward(layout.token_ids, layout.original_positions)
        self.assertGreater((faithful[-1] - shortened[-1]).abs().max().item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
