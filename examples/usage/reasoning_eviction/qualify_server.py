"""Compare warm and forced-cold Horizon sessions on an already-running server."""

import argparse
import json
import math
from pathlib import Path

from client import ReasoningSessionClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--logprob-atol", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--keep-going", action="store_true", help="Collect failures across windows"
    )
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    client = ReasoningSessionClient(
        args.url,
        AutoTokenizer.from_pretrained(
            args.tokenizer, revision=args.revision, trust_remote_code=False
        ),
    )
    reports = []
    failures = []
    for window in (None, 0, 1, 8, 4096):
        sessions = [client.open(window), client.open(window)]
        observed_drop = False
        try:
            for turn, messages in enumerate(
                (
                    [("user", "Compute 17 + 25. Keep the reasoning short.")],
                    [
                        ("tool", "The calculator confirms 42."),
                        ("user", "Add 3 to that result."),
                    ],
                    [("user", "What was the original sum, before adding 3?")],
                )
            ):
                if turn:
                    client.control(sessions[1], "evict")
                payload = client.build_turn(messages)
                results = [
                    client.generate(
                        sid,
                        **payload,
                        request_id=f"turn-{turn}",
                        sampling_params={
                            "temperature": 0,
                            "max_new_tokens": args.max_new_tokens,
                        },
                        return_logprob=True,
                    )
                    for sid in sessions
                ]
                stats = [
                    result["meta_info"]["reasoning_eviction"] for result in results
                ]
                if args.output_dir:
                    label = "full" if window is None else str(window)
                    (args.output_dir / f"w{label}-turn{turn}.json").write_text(
                        json.dumps({"warm": results[0], "cold": results[1]}, indent=2)
                        + "\n"
                    )
                observed_drop |= any(s["dropped_tokens"] > 0 for s in stats)
                if not all(s["completed"] for s in stats):
                    raise AssertionError(
                        "Qualification needs completed assistant messages; increase max-new-tokens"
                    )
                if results[0]["output_ids"] != results[1]["output_ids"]:
                    raise AssertionError(
                        f"Warm/cold greedy output differs: window={window}, turn={turn}"
                    )
                logs = [r["meta_info"]["output_token_logprobs"] for r in results]
                if not all(math.isfinite(entry[0]) for log in logs for entry in log):
                    raise AssertionError("Qualification produced non-finite logprobs")
                error = max(
                    (abs(a[0] - b[0]) for a, b in zip(*logs, strict=True)), default=0
                )
                if error > args.logprob_atol:
                    message = (
                        f"Warm/cold logprob difference {error} exceeds {args.logprob_atol}: "
                        f"window={window}, turn={turn}"
                    )
                    if not args.keep_going:
                        raise AssertionError(message)
                    failures.append(message)
                if turn and (stats[0]["replayed"] or not stats[1]["replayed"]):
                    raise AssertionError(
                        "Expected one warm and one replayed session; check cache capacity"
                    )
                retry = client.generate(
                    sessions[0],
                    **payload,
                    request_id=f"turn-{turn}",
                    sampling_params={
                        "temperature": 0,
                        "max_new_tokens": args.max_new_tokens,
                    },
                    return_logprob=True,
                )
                if retry != results[0]:
                    raise AssertionError("Completed retry changed its response")
                reports.append(
                    {
                        "window": window,
                        "turn": turn,
                        "max_logprob_error": error,
                        "stats": stats,
                    }
                )
            if window in (0, 1, 8) and not observed_drop:
                raise AssertionError(
                    f"Window {window} never pruned any tokens; use a longer reasoning trace"
                )
        except AssertionError as error:
            if not args.keep_going:
                raise
            failures.append(str(error))
        finally:
            for sid in sessions:
                client.control(sid, "close")
    print(
        json.dumps(
            {"passed": not failures, "failures": failures, "comparisons": reports},
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
