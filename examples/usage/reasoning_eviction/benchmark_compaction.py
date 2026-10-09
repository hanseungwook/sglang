"""Compare an explicitly supplied old implementation with current compaction.

Uses real BF16 CUDA tensors with Horizon-7B dimensions. Allocator/request
records are doubles: this measures the compaction operation, not full serving
latency. Source tensors and metadata are restored outside every timed trial.
Reports raw paired wall-clock samples, peak temporary allocation, source hashes,
and exact retained-KV verification. Run on an otherwise idle Hopper GPU.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import statistics
import time
from array import array
from pathlib import Path
from types import SimpleNamespace as NS

import torch

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    return dict(
        median=statistics.median(values), minimum=min(values), maximum=max(values)
    )


def benchmark_case(functions, prefix, reasoning, answer, page_size, repetitions):
    # prefix includes the opening marker; suffix includes closing and terminal.
    length = prefix + reasoning + answer + 2
    keep = [*range(prefix), *range(prefix + max(reasoning - 7, 0), length)]
    count = len(keep)
    capacity = (length + page_size - 1) // page_size * page_size
    pages = torch.randperm(capacity // page_size, device="cuda") + 2
    row = (
        pages[:, None] * page_size + torch.arange(page_size, device="cuda")
    ).flatten()
    shape = (capacity + 2 * page_size, 8, 128)
    # One working pool and one immutable pre-compaction snapshot.
    initial = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(72)
    ]
    buffers = [b.clone() for b in initial]
    pool = type("MHATokenToKVPool", (), {})()
    pool.k_buffer, pool.v_buffer = buffers[:36], buffers[36:]
    pool.kv_cache_layout, pool.use_hnd = "nhd", False
    kv = NS(
        cache_protected_len=0,
        cache_inserted_len=0,
        max_evicted_seqlen=0,
        holds_mamba=False,
        req_pool_idx=0,
    )
    req = NS(output_ids=[])
    session = NS(
        reasoning_owned=True,
        reasoning_cache_lost=False,
        _inflight=False,
        session_id="benchmark",
        req_nodes={"last": NS(req=req)},
    )
    freed = []
    cache = NS(
        session=NS(slots={"benchmark": NS(kv=kv, lock=None)}),
        token_to_kv_pool_allocator=NS(get_kvcache=lambda: pool),
        req_to_token_pool=NS(req_to_token=row[None]),
        page_size=page_size,
        free_kv_row=lambda record, ranges: freed.extend(ranges),
    )
    original_ids = array("q", range(length))

    def reset():
        for original, buf in zip(initial, buffers):
            buf.copy_(original)
        kv.kv_committed_len = kv.kv_allocated_len = length
        req.origin_input_ids = original_ids
        session.reasoning_position_offset = 137
        freed.clear()
        torch.cuda.synchronize()

    kwargs = dict(keep_indices=keep, expected_length=length, expected_offset=137)
    reference_stats = None
    reference_freed = None
    src, dst = row[row.new_tensor(keep)], row[:count]
    for name, function in functions.items():
        reset()
        stats = function(cache, session, **kwargs)
        torch.cuda.synchronize()
        for original, actual in zip(initial, buffers):
            assert torch.equal(actual[dst], original[src]), (name, "KV mismatch")
        assert list(req.origin_input_ids) == keep
        assert session.reasoning_position_offset + count == length + 137
        if reference_stats is None:
            reference_stats, reference_freed = stats, list(freed)
        else:
            assert stats == reference_stats and freed == reference_freed

    samples = {name: [] for name in functions}
    peaks = {name: [] for name in functions}
    order = list(functions)
    for repetition in range(repetitions + 2):
        # Counterbalance AB/BA order to reduce drift/clock bias. Warm two pairs.
        if repetition % 2:
            order = list(reversed(order))
        for name in order:
            reset()
            torch.cuda.reset_peak_memory_stats()
            before_bytes = torch.cuda.memory_allocated()
            start = time.perf_counter()
            functions[name](cache, session, **kwargs)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if repetition >= 2:
                samples[name].append(elapsed)
                peaks[name].append(torch.cuda.max_memory_allocated() - before_bytes)

    medians = {name: summary(values) for name, values in samples.items()}
    return dict(
        prefix_tokens=prefix,
        reasoning_tokens=reasoning,
        answer_tokens=answer,
        page_size=page_size,
        resident_before=length,
        resident_after=count,
        unchanged_prefix=prefix if reasoning > 7 else count,
        copied_tokens={
            "baseline": count if reasoning > 7 else 0,
            "optimized": count - prefix if reasoning > 7 else 0,
        },
        kv_bytes_per_token=36 * 2 * 8 * 128 * 2,
        exact_retained_kv=True,
        same_metadata_and_frees=True,
        wall_ms=medians,
        raw_wall_ms=samples,
        temporary_peak_bytes={k: summary(v) for k, v in peaks.items()},
        speedup=medians["baseline"]["median"] / medians["optimized"]["median"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ops", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    assert args.repetitions >= 3
    optimized_path = ROOT / "python/sglang/srt/session/reasoning_ops.py"
    baseline = load("baseline_compaction", args.baseline_ops)
    optimized = load("optimized_compaction", optimized_path)
    functions = {
        "baseline": baseline.compact_session,
        "optimized": optimized.compact_session,
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; CPU timings are not a substitute")
    torch.manual_seed(20261009)
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        layers=36,
        kv_heads=8,
        head_dim=128,
        dtype="bfloat16",
        baseline_source=str(args.baseline_ops),
        baseline_sha256=sha(args.baseline_ops),
        optimized_source=str(optimized_path),
        optimized_sha256=sha(optimized_path),
        repetitions=args.repetitions,
        warmup_pairs=2,
        scope="Compaction wall time, with allocator and request doubles; not end-to-end inference",
        cases=[],
    )
    prefixes = (4096,) if args.quick else (4096, 65536, 262144)
    for page in (1, 64):
        for prefix in prefixes:
            for reasoning, answer in ((0, 128), (512, 1024), (8192, 128)):
                result = benchmark_case(
                    functions, prefix, reasoning, answer, page, args.repetitions
                )
                report["cases"].append(result)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(result), flush=True)
                gc.collect()
                torch.cuda.empty_cache()
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
