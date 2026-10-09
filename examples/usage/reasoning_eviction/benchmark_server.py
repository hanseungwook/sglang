"""Old/new Horizon serving comparison with identical teacher-forced turns.

Synthetic reasoning and the answer are supplied, not generated: this measures
fixed-workload HTTP turn latency including prefill, a generated ending and compaction.
It does not predict the speed of long autoregressive reasoning trajectories.
Both servers run sequentially on the same allocated GPU with the same settings.
The separate generation suite tests real reasoning, interruption and replay.
"""

import argparse
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from client import ReasoningSessionClient


def synthetic_turn(tokenizer, prefix, reasoning):
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    tokens, messages = [], []

    def message(role, body, completed):
        start = len(tokens)
        tokens.extend(encode("<|ifm|im_start|>" + role))
        body_start = len(tokens)
        tokens.extend(encode("\n"))
        tokens.extend(body)
        if completed:
            tokens.extend(encode("<|ifm|im_end|>"))
        messages.append(
            dict(
                role=role,
                start=start,
                body_start=body_start,
                end=len(tokens),
                completed=completed,
            )
        )

    filler = encode(" Reference fact: seventeen plus twenty-five is forty-two.")
    body = (filler * (prefix // len(filler) + 1))[:prefix]
    message("user", body + encode("\nWhat is 17 + 25? Reply with only 42."), True)
    filler = encode(" The addition gives 42.")
    reasoning_ids = (filler * (reasoning // len(filler) + 1))[:reasoning]
    # The answer is also primed to make the remaining generation a short ending.
    message(
        "assistant",
        encode("<ifm|think>") + reasoning_ids + encode("</ifm|think>\n42"),
        False,
    )
    return dict(input_ids=tokens, messages=messages)


def trace(client, tokenizer, *, prefix, reasoning, turns, window=8, replay=False):
    sid = client.open(window)
    records = []
    try:
        for turn in range(turns):
            if replay and turn == 1:
                client.control(sid, "evict")
            payload = synthetic_turn(
                tokenizer, prefix if turn == 0 else 1024, reasoning
            )
            start = time.perf_counter()
            response = client.generate(
                sid,
                **payload,
                sampling_params={"temperature": 0, "max_new_tokens": 128},
                return_logprob=True,
            )
            elapsed = time.perf_counter() - start
            stats = response["meta_info"]["reasoning_eviction"]
            if not stats["completed"]:
                raise AssertionError(
                    "Synthetic primed answer did not finish; refusing a non-equivalent timing"
                )
            records.append(
                dict(
                    seconds=elapsed,
                    output_ids=response["output_ids"],
                    logprobs=response["meta_info"].get("output_token_logprobs"),
                    cache=stats,
                )
            )
    finally:
        client.control(sid, "close")
    return records


def generated_trace(client, *, window):
    """Generate real reasoning, testing a truncated continuation and cold replay."""
    sid = client.open(window)
    records = []
    previous_dropped = 0
    try:
        for turn, prompt in enumerate(
            ("What is 17 plus 25?", "Add 3 to that result.", "Double that result.")
        ):
            if turn == 2:
                client.control(sid, "evict")
            payload = client.build_turn([("user", prompt)])
            # Stop inside the current reasoning, then resume the same message.
            for segment in range(5):
                start = time.perf_counter()
                response = client.generate(
                    sid,
                    **payload,
                    sampling_params={
                        "temperature": 0,
                        "max_new_tokens": 1 if segment == 0 else 512,
                    },
                    return_logprob=True,
                )
                elapsed = time.perf_counter() - start
                stats = response["meta_info"]["reasoning_eviction"]
                records.append(
                    dict(
                        seconds=elapsed,
                        output_ids=response["output_ids"],
                        text=response.get("text"),
                        logprobs=response["meta_info"].get("output_token_logprobs"),
                        cache=stats,
                    )
                )
                if stats["completed"]:
                    previous_dropped = stats["dropped_tokens"]
                    break
                assert stats["dropped_tokens"] == previous_dropped, (
                    "Current incomplete reasoning was evicted"
                )
                payload = dict(input_ids=[], messages=[])
            else:
                raise AssertionError(
                    "Generated assistant did not finish in 2049 tokens"
                )
    finally:
        client.control(sid, "close")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkout", type=Path, required=True)
    parser.add_argument("--optimized-checkout", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--suite", choices=("timing", "generation"), default="timing")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    cases = (
        [(4096, 512, 4), (65536, 512, 16), (65536, 8192, 16), (262144, 8192, 4)]
        if args.suite == "timing"
        else []
    )
    report = dict(
        scope=(
            "Real 7B server; fixed teacher-forced reasoning and answer plus generated ending; not long decode throughput"
            if args.suite == "timing"
            else "Real generated 7B multi-turn reasoning; truncation and replay; correctness only"
        ),
        suite=args.suite,
        model=str(args.model),
        cases=cases,
        repetitions=args.repetitions,
        runs={},
    )
    for mode, checkout in [
        ("baseline", args.baseline_checkout),
        ("optimized", args.optimized_checkout),
    ]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(args.model),
            "--served-model-name",
            "IFM/K2-Horizon-7B",
            "--model-impl",
            "sglang",
            "--dtype",
            "bfloat16",
            "--attention-backend",
            "fa3",
            "--disable-overlap-schedule",
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--context-length",
            "524288",
            "--mem-fraction-static",
            "0.72",
            "--max-running-requests",
            "4",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--enable-streaming-session",
            "--enable-reasoning-kv-eviction",
            "--enable-deterministic-inference",
        ]
        env = dict(os.environ, PYTHONPATH=str(checkout / "python"))
        log = (args.output_dir / f"server-{mode}.log").open("w")
        (args.output_dir / f"command-{mode}.json").write_text(
            json.dumps(command, indent=2)
        )
        sources = {
            name: hashlib.sha256(
                (checkout / "python/sglang/srt/session" / name).read_bytes()
            ).hexdigest()
            for name in ["reasoning_ops.py", "reasoning_service.py"]
        }
        process = subprocess.Popen(
            command,
            cwd=checkout,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 900
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{mode} server exited; inspect log")
                try:
                    with urlopen(url + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup")
                time.sleep(1)
            client = ReasoningSessionClient(url, tokenizer, timeout=1800)
            # Warm both prefill and boundary paths before collecting samples.
            for _ in range(2):
                trace(client, tokenizer, prefix=2048, reasoning=256, turns=2)
            result = dict(sources=sources, timings={}, controls={})
            report["runs"][mode] = result
            for prefix, reasoning, turns in cases:
                key = f"p{prefix}-r{reasoning}-t{turns}"
                samples = []
                for repetition in range(args.repetitions):
                    records = trace(
                        client,
                        tokenizer,
                        prefix=prefix,
                        reasoning=reasoning,
                        turns=turns,
                    )
                    samples.append(records)
                    print(
                        json.dumps(
                            dict(
                                mode=mode,
                                case=key,
                                repetition=repetition,
                                seconds=sum(t["seconds"] for t in records),
                            )
                        ),
                        flush=True,
                    )
                    result["timings"][key] = samples
                    (args.output_dir / "server-comparison.json").write_text(
                        json.dumps(report, indent=2)
                    )
            # Full/W0/W1/W8, forced replay, and mixed-window batch smoke coverage.
            for window in (None, 0, 1, 8):
                result["controls"][str(window)] = trace(
                    client,
                    tokenizer,
                    prefix=4096,
                    reasoning=256,
                    turns=3,
                    window=window,
                    replay=True,
                )
            if args.suite == "generation":
                result["generated"] = {}
                for window in (None, 1, 8):
                    result["generated"][str(window)] = generated_trace(
                        client, window=window
                    )
                    print(
                        f"{mode}: generated full reasoning, window={window}", flush=True
                    )
                    (args.output_dir / "server-comparison.json").write_text(
                        json.dumps(report, indent=2)
                    )
            with ThreadPoolExecutor(max_workers=4) as executor:
                result["concurrent"] = list(
                    executor.map(
                        lambda window: trace(
                            client,
                            tokenizer,
                            prefix=2048,
                            reasoning=512,
                            turns=3,
                            window=window,
                        ),
                        (None, 1, 8, 8),
                    )
                )
            (args.output_dir / "server-comparison.json").write_text(
                json.dumps(report, indent=2)
            )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            log.close()

    def equivalent(a, b):
        # HTTP timings and server-generated IDs are excluded; actual output,
        # sampled probabilities, cache positions/counts and replay are checked.
        assert a["output_ids"] == b["output_ids"]
        assert a["cache"] == b["cache"]
        assert a["logprobs"] == b["logprobs"]

    before, after = report["runs"]["baseline"], report["runs"]["optimized"]
    report["summary"] = {}
    for key, old_samples in before["timings"].items():
        new_samples = after["timings"][key]
        for old, new in zip(old_samples, new_samples, strict=True):
            for a, b in zip(old, new, strict=True):
                equivalent(a, b)
        old_s = [sum(t["seconds"] for t in sample) for sample in old_samples]
        new_s = [sum(t["seconds"] for t in sample) for sample in new_samples]
        report["summary"][key] = dict(
            baseline_seconds=old_s,
            optimized_seconds=new_s,
            speedup=statistics.median(old_s) / statistics.median(new_s),
        )
    for key, old in before["controls"].items():
        for a, b in zip(old, after["controls"][key], strict=True):
            equivalent(a, b)
    for key, old in before.get("generated", {}).items():
        for a, b in zip(old, after["generated"][key], strict=True):
            equivalent(a, b)
    # Concurrent scheduling can change BF16 numerics; check cache invariants and
    # output tokens separately, without claiming bitwise batch logprob parity.
    for old, new in zip(before["concurrent"], after["concurrent"], strict=True):
        for a, b in zip(old, new, strict=True):
            assert a["output_ids"] == b["output_ids"]
            assert a["cache"] == b["cache"]
    report["passed"] = True
    (args.output_dir / "server-comparison.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
