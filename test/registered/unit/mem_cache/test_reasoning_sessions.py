"""CPU tests for the serving coordinator, actual KV moves and position handoff."""

import asyncio
import copy
import importlib.util
import os
import random
import sys
import tempfile
import unittest
from array import array
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load(
    "_reasoning_session_reference_tests",
    Path(__file__).with_name("test_reasoning_eviction.py"),
)
register_cpu_ci = base.register_cpu_ci
register_cpu_ci(est_time=8, suite="base-a-test-cpu")
with patch.dict(
    sys.modules, {"sglang.srt.mem_cache.reasoning_eviction": base.eviction}
):
    service_module = load(
        "_reasoning_service_test",
        ROOT / "python/sglang/srt/session/reasoning_service.py",
    )
ops = load("_reasoning_ops_test", ROOT / "python/sglang/srt/session/reasoning_ops.py")
config_module = load(
    "_reasoning_config_test", ROOT / "python/sglang/srt/session/reasoning_config.py"
)


class RecordingBackend:
    """Models native streaming's uncommitted last token, with observable forwards."""

    def __init__(self):
        self.sessions = {}
        self.outputs = []
        self.calls = []
        self.forwards = []
        self.block = None
        self.fail_once = False

    async def open(self, sid):
        self.sessions[sid] = dict(tokens=[], positions=[], offset=0)
        self.calls.append(("open", sid))

    async def close(self, sid):
        self.sessions.pop(sid, None)
        self.calls.append(("close", sid))

    async def acquire(self, sid):
        return {"resident": sid in self.sessions}

    async def park(self, sid):
        return {"committed_tokens": len(self.sessions[sid]["positions"])}

    def _forward(self, sid, new_tokens, reserve_last=False):
        state = self.sessions[sid]
        state["tokens"].extend(new_tokens)
        # Native prefix matching keeps a one-token logit reserve.
        start = min(len(state["positions"]), len(state["tokens"]) - 1)
        state["positions"] = state["positions"][:start]
        stop = len(state["tokens"]) - int(reserve_last)
        for index in range(start, stop):
            position = state["offset"] + index
            state["positions"].append(position)
            self.forwards.append((sid, position, tuple(state["positions"])))

    async def prefill(self, sid, tokens):
        self.calls.append(("prefill", list(tokens)))
        self._forward(sid, tokens)

    async def generate(self, sid, tokens, params):
        self.calls.append(("generate", list(tokens)))
        if self.block:
            await self.block.wait()
        if self.fail_once:
            self.fail_once = False
            raise service_module.ReplayRequired("cache was lost; replay required")
        output, finish = self.outputs.pop(0)
        self._forward(sid, tokens + output, reserve_last=True)
        return {
            "output_ids": list(output),
            "meta_info": {"finish_reason": {"type": finish}},
        }

    async def compact(self, sid, *, keep_indices, expected_length, expected_offset):
        self.calls.append(("compact", list(keep_indices)))
        state = self.sessions[sid]
        assert len(state["positions"]) == expected_length
        assert state["offset"] == expected_offset
        state["tokens"] = [state["tokens"][i] for i in keep_indices]
        state["positions"] = [state["positions"][i] for i in keep_indices]
        state["offset"] += expected_length - len(keep_indices)
        return {
            "freed_pages": (expected_length + 3) // 4 - (len(keep_indices) + 3) // 4
        }


def make_service(backend=None, **kwargs):
    tokenizer = base.Tokenizer()
    return service_module.ReasoningSessionService(
        backend or RecordingBackend(),
        markers=base.Markers.from_tokenizer(tokenizer),
        role_tokens={
            r: tokenizer.encode(r) for r in ("system", "user", "assistant", "tool")
        },
        identity="model-tokenizer-template-v1",
        context_length=1024,
        vocab_size=512,
        outside_tokens=(0, 1, tokenizer.ids["\n"]),
        **kwargs,
    )


def turns():
    _, tokens, messages, _, _ = base.fixture()
    result, previous = [], 0
    for index, message in enumerate(messages):
        if message.role != "assistant":
            continue
        prompt_end = message.body_start + 4  # newline, opener, two primed tokens
        new_messages = [m for m in messages[: index + 1] if m.start >= previous]
        new_messages[-1] = replace(message, end=prompt_end, completed=False)
        result.append(
            dict(
                input_ids=tokens[previous:prompt_end],
                messages=[
                    asdict(
                        replace(
                            m,
                            start=m.start - previous,
                            body_start=m.body_start - previous,
                            end=m.end - previous,
                        )
                    )
                    for m in new_messages
                ],
                output=tokens[prompt_end : message.end],
            )
        )
        previous = message.end
    return result


async def run_turn(service, sid, turn, rid="first"):
    service.backend.outputs.append((turn["output"], "stop"))
    return await service.generate(
        sid,
        request_id=rid,
        input_ids=turn["input_ids"],
        messages=turn["messages"],
        sampling_params={"max_new_tokens": 32},
    )


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_warm_tool_cycles_follow_reference_attention_graph(self):
        for window in (None, 0, 1, 8, 999):
            service = make_service()
            sid = service.open(window)["session_id"]
            for index, turn in enumerate(turns()):
                result = await run_turn(service, sid, turn, str(index))
                ref = service._reference(service.records[sid])
                for _, position, keys in service.backend.forwards:
                    self.assertEqual(keys, ref.visible_positions(position))
                layout = ref.layout_after(len(ref.token_ids))
                state = service.backend.sessions[service.resident[sid]]
                self.assertEqual(tuple(state["positions"]), layout.original_positions)
                stats = result["meta_info"]["reasoning_eviction"]
                self.assertEqual(stats["next_absolute_position"], len(ref.token_ids))
                self.assertEqual(stats["resident_kv_tokens"], len(layout.token_ids))
                self.assertTrue(stats["completed"])

    async def test_terminal_forward_precedes_compaction(self):
        service = make_service()
        sid = service.open(1)["session_id"]
        await run_turn(service, sid, turns()[0])
        calls = service.backend.calls
        index = next(i for i, call in enumerate(calls) if call[0] == "compact")
        self.assertEqual(calls[index - 1], ("prefill", []))

    async def test_forced_loss_replays_original_reasoning_before_continuing(self):
        for explicit in (True, False):
            service = make_service()
            sid = service.open(1)["session_id"]
            await run_turn(service, sid, turns()[0])
            if explicit:
                await service.evict(sid)
            else:
                service.backend.sessions.clear()  # idle allocator eviction
            service.backend.forwards.clear()
            result = await run_turn(service, sid, turns()[1], "second")
            ref = service._reference(service.records[sid])
            self.assertTrue(result["meta_info"]["reasoning_eviction"]["replayed"])
            for _, position, keys in service.backend.forwards:
                self.assertEqual(keys, ref.visible_positions(position))
            self.assertTrue(
                any(
                    position == ref.spans[0].start
                    for _, position, _ in service.backend.forwards
                )
            )

    async def test_retry_after_retraction_rolls_back_then_replays(self):
        service = make_service()
        sid = service.open(1)["session_id"]
        await run_turn(service, sid, turns()[0])
        service.backend.fail_once = True
        result = await run_turn(service, sid, turns()[1], "second")
        self.assertTrue(result["meta_info"]["reasoning_eviction"]["replayed"])
        self.assertEqual(len(service.records[sid]["results"]), 2)

    async def test_truncation_can_continue_without_pruning_current_reasoning(self):
        service = make_service()
        sid = service.open(0)["session_id"]
        turn = turns()[0]
        cut = 3
        service.backend.outputs.append((turn["output"][:cut], "length"))
        first = await service.generate(
            sid,
            request_id="part1",
            input_ids=turn["input_ids"],
            messages=turn["messages"],
            sampling_params={"max_new_tokens": cut},
        )
        self.assertFalse(first["meta_info"]["reasoning_eviction"]["completed"])
        self.assertEqual(service.records[sid]["events"], [])
        self.assertFalse(any(call[0] == "compact" for call in service.backend.calls))
        await service.evict(sid)
        service.backend.outputs.append((turn["output"][cut:], "stop"))
        second = await service.generate(
            sid,
            request_id="part2",
            input_ids=[],
            messages=[],
            sampling_params={"max_new_tokens": 32},
        )
        self.assertTrue(second["meta_info"]["reasoning_eviction"]["completed"])
        self.assertEqual(len(service.records[sid]["events"]), 1)
        ref = service._reference(service.records[sid])
        for _, position, keys in service.backend.forwards:
            self.assertEqual(keys, ref.visible_positions(position))

    async def test_retry_id_is_idempotent_and_conflicting_input_is_rejected(self):
        service = make_service()
        sid = service.open(8)["session_id"]
        turn = turns()[0]
        first = await run_turn(service, sid, turn)
        count = len(service.backend.calls)
        second = await service.generate(
            sid,
            request_id="first",
            input_ids=turn["input_ids"],
            messages=turn["messages"],
            sampling_params={"max_new_tokens": 32},
        )
        self.assertEqual(first, second)
        self.assertEqual(len(service.backend.calls), count)
        with self.assertRaises(service_module.SessionConflict):
            await service.generate(sid, request_id="first", input_ids=[], messages=[])

    async def test_disconnect_keeps_transaction_for_retry_and_serializes_session(self):
        service = make_service()
        sid = service.open(1)["session_id"]
        turn = turns()[0]
        service.backend.block = asyncio.Event()
        first = asyncio.create_task(run_turn(service, sid, turn))
        while not service.tasks:
            await asyncio.sleep(0)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        with self.assertRaises(service_module.SessionConflict):
            await service.generate(sid, request_id="other", input_ids=[], messages=[])
        with self.assertRaises(service_module.SessionConflict):
            await service.evict(sid)
        retry = asyncio.create_task(
            service.generate(
                sid,
                request_id="first",
                input_ids=turn["input_ids"],
                messages=turn["messages"],
                sampling_params={"max_new_tokens": 32},
            )
        )
        service.backend.block.set()
        result = await retry
        self.assertEqual(result["output_ids"], turn["output"])
        self.assertEqual(sum(c[0] == "generate" for c in service.backend.calls), 1)

    async def test_explicit_cancellation_rolls_back_and_releases_backend(self):
        service = make_service()
        sid = service.open(1)["session_id"]
        service.backend.block = asyncio.Event()
        task = asyncio.create_task(run_turn(service, sid, turns()[0]))
        while not any(c[0] == "generate" for c in service.backend.calls):
            await asyncio.sleep(0)
        self.assertTrue((await service.cancel(sid))["rolled_back"])
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(service.records[sid]["token_ids"], [])
        self.assertEqual(service.backend.sessions, {})
        self.assertNotIn(sid, service.resident)

    async def test_persistence_restores_events_retry_results_and_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(store=directory)
            sid = service.open(8)["session_id"]
            first = await run_turn(service, sid, turns()[0])
            service._store_lock.close()  # simulate process exit before restart
            restored = make_service(store=directory)
            turn = turns()[0]
            cached = await restored.generate(
                sid,
                request_id="first",
                input_ids=turn["input_ids"],
                messages=turn["messages"],
                sampling_params={"max_new_tokens": 32},
            )
            self.assertEqual(cached, first)
            self.assertEqual(restored.backend.calls, [])
            second = await run_turn(restored, sid, turns()[1], "second")
            self.assertTrue(second["meta_info"]["reasoning_eviction"]["replayed"])
            self.assertEqual(len(restored.records[sid]["events"]), 2)
            restored._store_lock.close()

    async def test_store_rejects_concurrent_writer_and_incompatible_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(store=directory)
            sid = service.open()["session_id"]
            with self.assertRaisesRegex(ValueError, "already in use"):
                make_service(store=directory)
            service._store_lock.close()
            restored = make_service(store=directory)
            restored.identity = "different-model"
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                restored._get(sid)
            restored._store_lock.close()

    async def test_failed_persistence_does_not_commit_a_response(self):
        service = make_service()
        sid = service.open(1)["session_id"]
        with patch.object(service, "_save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                await run_turn(service, sid, turns()[0])
        self.assertEqual(service.records[sid]["token_ids"], [])
        self.assertEqual(service.backend.sessions, {})

    async def test_later_failed_transaction_preserves_history_and_retry_results(self):
        service = make_service()
        sid = service.open(8)["session_id"]
        first = await run_turn(service, sid, turns()[0])
        committed = copy.deepcopy(service.records[sid])
        # Mutating a caller's response must not mutate the cached retry value.
        first["output_ids"].clear()
        first["meta_info"]["reasoning_eviction"]["window"] = 999
        with patch.object(service, "_save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                await run_turn(service, sid, turns()[1], "failed-second")
        self.assertEqual(service.records[sid], committed)
        turn = turns()[0]
        retry = await service.generate(
            sid,
            request_id="first",
            input_ids=turn["input_ids"],
            messages=turn["messages"],
            sampling_params={"max_new_tokens": 32},
        )
        self.assertEqual(retry, committed["results"]["first"]["response"])
        await run_turn(service, sid, turns()[1], "second")
        self.assertEqual(
            committed["results"]["first"], service.records[sid]["results"]["first"]
        )
        self.assertEqual(len(service.records[sid]["results"]), 2)

    async def test_failed_incomplete_continuation_preserves_previous_boundary(self):
        service = make_service()
        sid = service.open(1)["session_id"]
        turn = turns()[0]
        service.backend.outputs.append((turn["output"][:3], "length"))
        await service.generate(
            sid,
            request_id="partial",
            input_ids=turn["input_ids"],
            messages=turn["messages"],
            sampling_params={"max_new_tokens": 3},
        )
        committed = copy.deepcopy(service.records[sid])
        service.backend.outputs.append((turn["output"][3:], "stop"))
        with patch.object(service, "_save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                await service.generate(
                    sid, request_id="finish", input_ids=[], messages=[]
                )
        self.assertEqual(service.records[sid], committed)
        self.assertFalse(service.records[sid]["messages"][-1]["completed"])

    async def test_context_limit_uses_full_history(self):
        service = make_service()
        sid = service.open(0)["session_id"]
        await run_turn(service, sid, turns()[0])
        service.context_length = (
            len(service.records[sid]["token_ids"]) + len(turns()[1]["input_ids"]) + 32
        )
        with self.assertRaisesRegex(ValueError, "original context"):
            await run_turn(service, sid, turns()[1], "second")

    async def test_bad_roles_and_request_shapes_fail_before_backend_calls(self):
        service = make_service()
        sid = service.open(1)["session_id"]
        turn = turns()[0]
        bad_messages = copy.deepcopy(turn["messages"])
        bad_messages[0]["role"] = "tool"
        for ids, messages in (
            (turn["input_ids"], bad_messages),
            ([-1], []),
            (turn["input_ids"], []),
        ):
            with self.assertRaises(ValueError):
                await service.generate(
                    sid, request_id="bad", input_ids=ids, messages=messages
                )
        self.assertEqual(service.backend.calls, [])


class PoolTests(unittest.TestCase):
    def pool_fixture(self, length, page_size):
        device = os.environ.get("REASONING_KV_TEST_DEVICE", "cpu")
        allocated = (length + page_size - 1) // page_size * page_size
        # Use a nontrivial page mapping to catch treating logical indices as slots.
        page_order = list(range(allocated // page_size + 1, 1, -1))
        slots = [page * page_size + j for page in page_order for j in range(page_size)]
        row = torch.tensor([slots + [0] * 20], device=device)
        size = max(slots) + 1
        pool = type("MHATokenToKVPool", (), {})()
        pool.use_hnd = False
        pool.k_buffer = [
            torch.arange(size * 4, device=device).reshape(size, 2, 2).to(torch.bfloat16)
            + i * 10
            for i in range(3)
        ]
        pool.v_buffer = [buf + 100 for buf in pool.k_buffer]
        pool.kv_cache_layout = "nhd"
        kv = SimpleNamespace(
            cache_protected_len=0,
            cache_inserted_len=0,
            max_evicted_seqlen=0,
            holds_mamba=False,
            kv_committed_len=length,
            kv_allocated_len=length,
            req_pool_idx=0,
        )
        req = SimpleNamespace(origin_input_ids=array("q", range(length)), output_ids=[])
        session = SimpleNamespace(
            reasoning_owned=True,
            reasoning_cache_lost=False,
            reasoning_position_offset=9,
            _inflight=False,
            session_id="test",
            req_nodes={"last": SimpleNamespace(req=req)},
        )
        freed = []

        def free_kv_row(record, ranges):
            for start, stop in ranges:
                freed.extend(set(int(slot) // page_size for slot in row[0, start:stop]))

        cache = SimpleNamespace(
            session=SimpleNamespace(slots={"test": SimpleNamespace(kv=kv, lock=None)}),
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: pool),
            req_to_token_pool=SimpleNamespace(req_to_token=row),
            page_size=page_size,
            free_kv_row=free_kv_row,
        )
        return cache, session, pool, slots, freed

    def test_actual_moves_preserve_every_layer_and_free_only_whole_unused_pages(self):
        for page in (1, 4, 16):
            for length in (17, 31, 32, 33):
                cache, session, pool, slots, freed = self.pool_fixture(length, page)
                keep = [0, 1, *range(length - 7, length)]
                before = [b.clone() for b in (*pool.k_buffer, *pool.v_buffer)]
                result = ops.compact_session(
                    cache,
                    session,
                    keep_indices=keep,
                    expected_length=length,
                    expected_offset=9,
                )
                for original, buf in zip(before, (*pool.k_buffer, *pool.v_buffer)):
                    torch.testing.assert_close(
                        buf[slots[: len(keep)]],
                        original[[slots[i] for i in keep]],
                        rtol=0,
                        atol=0,
                    )
                self.assertEqual(
                    list(session.req_nodes["last"].req.origin_input_ids), keep
                )
                self.assertEqual(result["next_absolute_position"], length + 9)
                self.assertEqual(len(freed), len(set(freed)))
                retained_pages = {slot // page for slot in slots[: len(keep)]}
                self.assertFalse(retained_pages.intersection(freed))
                self.assertEqual(len(freed), result["freed_pages"])

    def test_refuses_uncommitted_shared_stale_and_invalid_compaction(self):
        for kind in (
            "uncommitted",
            "shared",
            "stale",
            "missing_terminal",
            "duplicate",
            "inflight",
            "negative",
            "unordered",
            "bool",
            "float",
            "out_of_range",
            "empty",
        ):
            cache, session, pool, _, freed = self.pool_fixture(17, 4)
            kv = cache.session.slots["test"].kv
            keep = [0, 16]
            offset = 9
            if kind == "uncommitted":
                kv.kv_committed_len -= 1
            if kind == "shared":
                kv.cache_protected_len = 1
            if kind == "stale":
                offset = 0
            if kind == "missing_terminal":
                keep = [0, 15]
            if kind == "duplicate":
                keep = [0, 0, 16]
            if kind == "inflight":
                session._inflight = True
            keep = {
                "negative": [-1, 16],
                "unordered": [4, 1, 16],
                "bool": [False, 16],
                "float": [0, 16.0],
                "out_of_range": [0, 18, 16],
                "empty": [],
            }.get(kind, keep)
            before = pool.k_buffer[0].clone()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                ops.compact_session(
                    cache,
                    session,
                    keep_indices=keep,
                    expected_length=17,
                    expected_offset=offset,
                )
            torch.testing.assert_close(pool.k_buffer[0], before)
            self.assertEqual(freed, [])

    def test_repeated_random_compactions_match_full_gather_bitwise(self):
        rng = random.Random(20261009)
        for page in (1, 4, 64):
            for trial in range(12):
                length = rng.randrange(65, 160)
                cache, session, pool, slots, freed = self.pool_fixture(length, page)
                token_ids = list(range(length))
                for step in range(4):
                    prefix = rng.randrange(length)
                    keep = [
                        *range(prefix),
                        *[i for i in range(prefix, length - 1) if rng.random() < 0.55],
                        length - 1,
                    ]
                    original = [b.clone() for b in (*pool.k_buffer, *pool.v_buffer)]
                    previous_offset = session.reasoning_position_offset
                    result = ops.compact_session(
                        cache,
                        session,
                        keep_indices=keep,
                        expected_length=length,
                        expected_offset=previous_offset,
                    )
                    for before, buf in zip(original, (*pool.k_buffer, *pool.v_buffer)):
                        expected = before.clone()
                        expected[slots[: len(keep)]] = before[[slots[i] for i in keep]]
                        torch.testing.assert_close(buf, expected, rtol=0, atol=0)
                    token_ids = [token_ids[i] for i in keep]
                    self.assertEqual(
                        list(session.req_nodes["last"].req.origin_input_ids), token_ids
                    )
                    self.assertEqual(
                        result["next_absolute_position"], length + previous_offset
                    )
                    self.assertEqual(len(freed), len(set(freed)))
                    self.assertFalse(
                        set(freed) & {slot // page for slot in slots[: len(keep)]}
                    )
                    length = len(keep)

    def test_unchanged_prefix_is_not_read_or_written_and_noop_never_touches_gpu(self):
        class ObservedBuffer:
            def __init__(self, tensor):
                self.tensor = tensor
                self.dtype, self.ndim = tensor.dtype, tensor.ndim
                self.reads, self.writes = [], []

            def __getitem__(self, indices):
                self.reads.extend(indices.tolist())
                return self.tensor[indices]

            def __setitem__(self, indices, values):
                self.writes.extend(indices.tolist())
                self.tensor[indices] = values

        for noop in (False, True):
            cache, session, pool, slots, _ = self.pool_fixture(100, 4)
            pool.k_buffer = [ObservedBuffer(b) for b in pool.k_buffer]
            pool.v_buffer = [ObservedBuffer(b) for b in pool.v_buffer]
            keep = list(range(100)) if noop else [*range(90), 97, 98, 99]
            if noop:
                with patch.object(
                    torch.cuda,
                    "synchronize",
                    side_effect=AssertionError("No-op synchronized GPU"),
                ):
                    ops.compact_session(
                        cache,
                        session,
                        keep_indices=keep,
                        expected_length=100,
                        expected_offset=9,
                    )
            else:
                ops.compact_session(
                    cache,
                    session,
                    keep_indices=keep,
                    expected_length=100,
                    expected_offset=9,
                )
            for buf in (*pool.k_buffer, *pool.v_buffer):
                self.assertEqual(
                    buf.reads, [] if noop else [slots[i] for i in (97, 98, 99)]
                )
                self.assertEqual(buf.writes, [] if noop else slots[90:93])

    def test_positions_follow_filtered_merged_and_chunked_request_order(self):
        a, b, normal = [
            SimpleNamespace(reasoning_position_offset=i) for i in (4, 19, 0)
        ]
        self.assertEqual(
            ops.apply_reasoning_positions(torch.tensor([7, 12]), [a, b]).tolist(),
            [11, 31],
        )
        self.assertEqual(
            ops.apply_reasoning_positions(torch.tensor([12]), [b]).tolist(), [31]
        )
        self.assertEqual(
            ops.apply_reasoning_positions(
                torch.tensor([12, 3, 7]), [b, normal, a]
            ).tolist(),
            [31, 3, 11],
        )
        self.assertEqual(
            ops.apply_reasoning_positions(
                torch.tensor([5, 6, 7, 10, 11]), [a, b], [3, 2]
            ).tolist(),
            [9, 10, 11, 29, 30],
        )


class ConfigurationTests(unittest.TestCase):
    def config(self):
        return SimpleNamespace(
            enable_reasoning_kv_eviction=True,
            enable_streaming_session=True,
            disable_overlap_schedule=True,
            tokenizer_worker_num=1,
            tp_size=1,
            pp_size=1,
            dp_size=1,
            dcp_size=1,
            attention_backend="triton",
            disaggregation_mode="null",
            disable_radix_cache=False,
            speculative_algorithm=None,
            quantization=None,
            dtype="bfloat16",
            kv_cache_dtype="auto",
            model_impl="auto",
            device="cuda",
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend="disabled"),
                prefill=SimpleNamespace(backend="disabled"),
            ),
        )

    def test_supported_config_and_default_off(self):
        for backend in ("triton", "fa3"):
            for explicit in (False, True):
                cfg = self.config()
                cfg.attention_backend = backend
                if explicit:
                    cfg.prefill_attention_backend = backend
                    cfg.decode_attention_backend = backend
                with self.subTest(backend=backend, explicit=explicit):
                    config_module.validate_reasoning_eviction_config(cfg)
        config_module.validate_reasoning_eviction_config(
            SimpleNamespace(enable_reasoning_kv_eviction=False)
        )

    def test_mixed_attention_backends_fail_closed(self):
        for backend in ("triton", "fa3"):
            for phase in ("prefill_attention_backend", "decode_attention_backend"):
                for other in ("triton", "fa3", "flashinfer", "fa4"):
                    if other == backend:
                        continue
                    cfg = self.config()
                    cfg.attention_backend = backend
                    setattr(cfg, phase, other)
                    with (
                        self.subTest(backend=backend, phase=phase, other=other),
                        self.assertRaisesRegex(ValueError, phase),
                    ):
                        config_module.validate_reasoning_eviction_config(cfg)

    def test_unsupported_configs_fail_closed(self):
        for name, value in (
            ("tp_size", 2),
            ("disable_overlap_schedule", False),
            ("enable_hierarchical_cache", True),
            ("quantization", "fp8"),
            ("dtype", "float16"),
            ("kv_cache_dtype", "fp8_e4m3"),
            ("attention_backend", "flashinfer"),
            ("tokenizer_worker_num", 2),
            ("speculative_algorithm", "EAGLE"),
            ("cuda_graph_config", None),
        ):
            cfg = self.config()
            setattr(cfg, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                config_module.validate_reasoning_eviction_config(cfg)


class StreamingHookTests(unittest.TestCase):
    """Run the actual streaming-session module with cache/scheduler test doubles."""

    def setUp(self):
        class Record(SimpleNamespace):
            def __init__(self, **kwargs):
                super().__init__(
                    req_pool_idx=None,
                    kv_allocated_len=0,
                    kv_committed_len=0,
                    cache_protected_len=0,
                    max_evicted_seqlen=0,
                )
                self.__dict__.update(kwargs)

            @property
            def holds_kv(self):
                return self.req_pool_idx is not None

            def clamp_evicted_seqlens(self, length):
                pass  # This double models only full attention.

        class Abort:
            def __init__(self, message, status_code=400):
                self.message, self.status_code = message, status_code

        self.Record, self.Abort = Record, Abort
        dependencies = {
            "sglang.srt.managers.schedule_batch": SimpleNamespace(
                ReqKvInfo=Record, FINISH_ABORT=Abort
            ),
            "sglang.srt.mem_cache.base_prefix_cache": SimpleNamespace(
                MatchPrefixParams=SimpleNamespace,
                MatchResult=SimpleNamespace,
                TreeLock=object,
            ),
            "sglang.srt.utils.common": SimpleNamespace(
                ceil_align=lambda n, page: (n + page - 1) // page * page,
                is_npu=lambda: False,
            ),
        }
        patcher = patch.dict(sys.modules, dependencies)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.module = load(
            "_reasoning_streaming_hooks_test",
            ROOT / "python/sglang/srt/session/streaming_session.py",
        )
        self.freed, self.published, self.finished, self.aborted = [], [], [], []

        def free_row(slot):
            slot.kv.req_pool_idx = None

        self.cache = SimpleNamespace(
            page_size=4,
            tree_core=SimpleNamespace(
                empty_match_result=SimpleNamespace(
                    device_indices=torch.empty(0, dtype=torch.int64)
                )
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(64).reshape(1, 64), free=free_row
            ),
            unlock=lambda lock: None,
            free_kv_row=lambda kv, ranges: self.freed.extend(ranges),
            checkpoint_into_tree=lambda *args, **kwargs: self.published.append(kwargs),
        )
        self.streaming = self.module.StreamingSession(self.cache)
        self.cache.session = self.streaming

    def request(self, sid="reasoning", owned=True):
        owner = SimpleNamespace(
            session_id=sid,
            streaming=True,
            reasoning_owned=owned,
            reasoning_busy=False,
            reasoning_cache_lost=False,
            _inflight=False,
            finish_req=self.finished.append,
            abort_req=lambda: self.aborted.append(sid),
        )
        req = SimpleNamespace(
            session=owner,
            rid=sid,
            lock=None,
            last_node=None,
            to_finish=None,
            finished_reason=None,
            finished_len=None,
            origin_input_ids=list(range(6)),
            output_ids=[],
            swa_branching_seqlen=None,
            kv=self.Record(req_pool_idx=0, kv_allocated_len=6, kv_committed_len=6),
        )
        req.finished = lambda: req.finished_reason is not None

        def detach():
            kv, req.kv = req.kv, self.Record()
            return kv

        req.detach_kv = detach
        return req

    def test_first_match_and_checkpoints_never_publish_reasoning(self):
        req = self.request()
        match = self.streaming.try_match_prefix(SimpleNamespace(req=req))
        self.assertEqual(match.device_indices.numel(), 0)
        self.assertTrue(self.module.is_virtual_node(match.last_device_node))
        self.streaming.take(req)
        self.assertFalse(self.streaming.slots["reasoning"].publishes_prompt)
        self.assertTrue(self.streaming.try_checkpoint(req, up_to=4))
        self.assertEqual(req.prefix_indices.tolist(), [0, 1, 2, 3])
        self.assertEqual(self.published, [])

    def test_successful_finish_transfers_ownership_without_freeing(self):
        req = self.request()
        kv = req.kv
        self.streaming.take(req)
        req.finished_reason = object()
        self.assertTrue(self.streaming.try_cache_finished_req(req))
        self.assertFalse(req.kv.holds_kv)
        self.assertIs(self.streaming.slots["reasoning"].kv, kv)
        self.assertEqual(self.finished, [req])
        self.assertEqual(self.freed, [])

    def test_retraction_and_abort_release_once_and_require_replay(self):
        for abort in (False, True):
            with self.subTest(abort=abort):
                req = self.request()
                self.streaming.take(req)
                if abort:
                    req.finished_reason = self.Abort("cancelled")
                self.assertTrue(self.streaming.try_cache_finished_req(req))
                self.assertTrue(req.session.reasoning_cache_lost)
                self.assertFalse(req.kv.holds_kv)
                self.assertNotIn("reasoning", self.streaming.slots)
                if not abort:
                    self.assertEqual(req.to_finish.status_code, 503)
                before = list(self.freed)
                self.streaming.release_session("reasoning")
                self.assertEqual(self.freed, before)
        self.assertEqual(self.freed, [(0, 6), (0, 6)])

    def test_busy_or_inflight_sessions_are_not_pressure_eviction_candidates(self):
        req = self.request()
        self.streaming.take(req)
        for flag in ("reasoning_busy", "_inflight"):
            setattr(req.session, flag, True)
            self.assertEqual(self.streaming.idle_reasoning_size(), 0)
            self.assertEqual(self.streaming.evict_idle_reasoning_sessions(4), 0)
            setattr(req.session, flag, False)
        self.assertEqual(self.streaming.idle_reasoning_size(), 8)
        self.assertEqual(self.streaming.evict_idle_reasoning_sessions(4), 8)
        self.assertTrue(req.session.reasoning_cache_lost)
        self.assertEqual(self.streaming.idle_reasoning_size(), 0)
        self.assertEqual(self.freed, [(0, 6)])

    def test_ordinary_streaming_session_keeps_existing_publish_and_abort_behavior(self):
        req = self.request(owned=False)
        self.assertIsNone(self.streaming.try_match_prefix(SimpleNamespace(req=req)))
        self.streaming.take(req)
        self.assertTrue(self.streaming.slots["reasoning"].publishes_prompt)
        self.streaming.try_checkpoint(req, up_to=4)
        self.assertEqual(len(self.published), 1)
        self.assertEqual(self.streaming.idle_reasoning_size(), 0)
        req.finished_reason = self.Abort("cancelled")
        self.assertFalse(self.streaming.try_cache_finished_req(req))
        self.assertTrue(req.kv.holds_kv)
        self.assertEqual(self.freed, [])

    def test_row_pressure_releases_idle_rows_and_continuations_borrow_their_own(self):
        active = self.request("active")
        self.streaming.take(active)
        active.session.reasoning_busy = True
        active.detach_kv()
        self.assertTrue(ops.has_reasoning_session_row(self.cache, active))
        self.assertEqual(ops.prefill_row_cost(self.cache, active), 0)
        for sid in ("idle-1", "idle-2"):
            self.streaming.take(self.request(sid))
        new = self.request("new")
        new.detach_kv()
        self.assertEqual(ops.prefill_row_cost(self.cache, new), 1)
        self.assertEqual(
            self.streaming.evict_idle_reasoning_sessions(0, num_rows=2), 16
        )
        self.assertEqual(list(self.streaming.slots), ["active"])
        self.assertEqual(self.streaming.evict_idle_reasoning_sessions(0), 0)


if __name__ == "__main__":
    unittest.main()
