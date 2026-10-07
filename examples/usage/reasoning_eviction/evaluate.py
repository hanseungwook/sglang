"""Evaluate a token trace's reasoning eviction policy without a serving install."""

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path


def load_reference():
    # Keep the reference evaluator runnable without SGLang's GPU dependencies.
    root = Path(__file__).resolve().parents[3]
    source = root / "python/sglang/srt/mem_cache/reasoning_eviction.py"
    spec = importlib.util.spec_from_file_location(
        "_reasoning_eviction_reference", source
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_window(value):
    if value == "full":
        return None
    try:
        window = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "window must be full or a nonnegative integer"
        ) from error
    if window < 0:
        raise argparse.ArgumentTypeError("window must be full or a nonnegative integer")
    return window


def evaluate(trace, window, chunk_tokens=None):
    module = load_reference()
    tokens = trace["token_ids"]
    markers = module.IFMReasoningMarkers(**trace["markers"])
    messages = tuple(module.MessageBoundary(**message) for message in trace["messages"])
    spans = module.annotate_reasoning(tokens, messages, markers)
    policy = module.ReasoningEvictionPolicy(window)
    reference = module.ReasoningEvictionReference(tokens, spans, policy)
    layout = reference.layout_after(len(tokens))
    boundaries = []
    for boundary in sorted({span.expires_at for span in spans}):
        before = reference.visible_positions(boundary - 1)
        plan = reference.plan_compaction(boundary, before, committed_through=boundary)
        boundaries.append(
            {
                "next_absolute_position": boundary,
                "reasoning_tokens": sum(
                    s.stop - s.start for s in spans if s.expires_at == boundary
                ),
                "dropped_at_boundary": len(plan.drop_indices),
                "resident_tokens": len(plan.layout.token_ids),
                "cumulative_dropped_tokens": boundary - len(plan.layout.token_ids),
            }
        )
    return {
        "mode": "cpu_reference_only",
        "policy": asdict(policy),
        "transcript_tokens": len(tokens),
        "reasoning_tokens": sum(span.stop - span.start for span in spans),
        "retained_tokens": len(layout.token_ids),
        "dropped_tokens": len(tokens) - len(layout.token_ids),
        "next_absolute_position": layout.next_absolute_position,
        "retained_original_positions": layout.original_positions,
        "reasoning_spans": [asdict(span) for span in spans],
        "boundaries": boundaries,
        "replay_steps": [asdict(step) for step in reference.replay_steps(chunk_tokens)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace", type=Path, help="JSON with token_ids, markers and trusted messages"
    )
    parser.add_argument(
        "--window", type=parse_window, default=None, help="full (default), 0, 1, 8, ..."
    )
    parser.add_argument(
        "--chunk-tokens", type=int, help="maximum teacher-forcing chunk size"
    )
    args = parser.parse_args()
    try:
        report = evaluate(
            json.loads(args.trace.read_text()), args.window, args.chunk_tokens
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
