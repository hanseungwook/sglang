"""Measure the two transaction clones separately from parsing and inference."""

import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load(
        "sglang.srt.mem_cache.reasoning_eviction",
        ROOT / "python/sglang/srt/mem_cache/reasoning_eviction.py",
    )
    source = ROOT / "python/sglang/srt/session/reasoning_service.py"
    service = load("metadata_service", source)
    report = dict(
        scope="Two transaction clones only; excludes annotation, IPC, GPU and persistence",
        python=platform.python_version(),
        host=platform.node(),
        optimized_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        cases=[],
    )
    for tokens in (10000, 100000, 500000):
        turns = max(1, tokens // 5000)
        record = dict(
            token_ids=list(range(tokens)),
            messages=[
                dict(
                    role="assistant",
                    start=i * 5000,
                    body_start=i * 5000 + 2,
                    end=min(tokens, (i + 1) * 5000),
                    completed=True,
                )
                for i in range(turns)
            ],
            results={
                str(i): {
                    "digest": "abc",
                    "response": {
                        "output_ids": list(range(4096)),
                        "meta_info": {"reasoning_eviction": {"window": 8}},
                    },
                }
                for i in range(turns)
            },
            events=[],
            policy={"window": 8},
            identity="fixed",
        )
        functions = dict(
            baseline=copy.deepcopy, optimized=service.copy_record_for_append
        )
        samples = {key: [] for key in functions}
        for repetition in range(11):
            order = (
                list(functions) if repetition % 2 == 0 else list(reversed(functions))
            )
            for name in order:
                start = time.perf_counter()
                candidate = functions[name](record)
                attempt = functions[name](candidate)
                elapsed = (time.perf_counter() - start) * 1000
                assert candidate == record and attempt == record
                if repetition >= 2:
                    samples[name].append(elapsed)
                del attempt, candidate
        before, after = [statistics.median(samples[k]) for k in functions]
        report["cases"].append(
            dict(
                transcript_tokens=tokens,
                cached_responses=turns,
                raw_wall_ms=samples,
                baseline_ms=before,
                optimized_ms=after,
                speedup=before / after,
            )
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
