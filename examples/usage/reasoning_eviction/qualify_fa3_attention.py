"""Check real FA3 attention after KV compaction against a dense masked oracle.

Requires a Hopper GPU. Allocation records are test doubles, but KV copies,
position offsets, and FA3 kernels are the serving implementations. This checks
attention outputs, not full-model logits or model quality.
"""

import argparse
import json
import math
from array import array
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.srt.session.reasoning_ops import (
    apply_reasoning_positions,
    compact_session,
)


def rotary(x, positions):
    dim = x.shape[-1]
    frequencies = 10000 ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
    angles = positions.float()[:, None, None] * frequencies
    a, b = x.float().chunk(2, dim=-1)
    return torch.cat(
        (a * angles.cos() - b * angles.sin(), b * angles.cos() + a * angles.sin()),
        dim=-1,
    ).to(x.dtype)


def check(length, page_size, window, query_length, head_dim):
    device = "cuda"
    query_heads, kv_heads, offset = 8, 2, 137
    capacity = math.ceil((length + query_length) / page_size) * page_size
    page_order = torch.randperm(capacity // page_size, device=device) + 2
    row = (
        page_order[:, None] * page_size + torch.arange(page_size, device=device)
    ).flatten()
    size = capacity + 2 * page_size
    shape = (length + query_length, kv_heads, head_dim)
    keys = rotary(
        torch.randn(shape, device=device, dtype=torch.bfloat16),
        torch.arange(offset, offset + shape[0], device=device),
    )
    values = torch.randn(shape, device=device, dtype=torch.bfloat16)
    pool = type("MHATokenToKVPool", (), {})()
    pool.use_hnd, pool.kv_cache_layout = False, "nhd"
    pool.k_buffer = [
        torch.randn(size, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    ]
    pool.v_buffer = [torch.randn_like(pool.k_buffer[0])]
    pool.k_buffer[0][row[:length]] = keys[:length]
    pool.v_buffer[0][row[:length]] = values[:length]
    kv = NS(
        cache_protected_len=0,
        cache_inserted_len=0,
        max_evicted_seqlen=0,
        holds_mamba=False,
        kv_committed_len=length,
        kv_allocated_len=length,
        req_pool_idx=0,
    )
    req = NS(origin_input_ids=array("q", range(length)), output_ids=[])
    session = NS(
        reasoning_owned=True,
        reasoning_cache_lost=False,
        reasoning_position_offset=offset,
        _inflight=False,
        session_id="test",
        req_nodes={"last": NS(req=req)},
    )
    freed = []
    cache = NS(
        session=NS(slots={"test": NS(kv=kv, lock=None)}),
        token_to_kv_pool_allocator=NS(get_kvcache=lambda: pool),
        req_to_token_pool=NS(req_to_token=row[None]),
        page_size=page_size,
        free_kv_row=lambda record, ranges: freed.extend(ranges),
    )
    # Prefix + opening marker, optional reasoning tail, closing marker + answer.
    keep = (
        list(range(length))
        if window is None
        else [*range(17), *range(length - 17 - (window - 1), length)]
    )
    result = compact_session(
        cache,
        session,
        keep_indices=keep,
        expected_length=length,
        expected_offset=offset,
    )
    retained = len(keep)
    assert result["next_absolute_position"] == length + offset
    # Simulate appending into the owned tail page and newly allocated pages.
    pool.k_buffer[0][row[retained : retained + query_length]] = keys[length:]
    pool.v_buffer[0][row[retained : retained + query_length]] = values[length:]
    positions = apply_reasoning_positions(
        torch.arange(retained, retained + query_length, device=device),
        [session],
        [query_length],
    )
    torch.testing.assert_close(
        positions,
        torch.arange(length + offset, length + offset + query_length, device=device),
    )
    queries = rotary(
        torch.randn(
            query_length, query_heads, head_dim, device=device, dtype=torch.bfloat16
        ),
        positions,
    )
    # Full chronological K/V, with evicted keys and future tokens masked out.
    visible = torch.zeros(length + query_length, device=device, dtype=torch.bool)
    visible[keep] = True
    visible[length:] = True
    key_positions = torch.arange(length + query_length, device=device) + offset
    mask = visible[None, :] & (key_positions[None, :] <= positions[:, None])
    expanded_keys = keys.float().repeat_interleave(query_heads // kv_heads, dim=1)
    expanded_values = values.float().repeat_interleave(query_heads // kv_heads, dim=1)
    scores = torch.einsum("qhd,khd->hqk", queries.float(), expanded_keys) / math.sqrt(
        head_dim
    )
    probabilities = scores.masked_fill(~mask[None], float("-inf")).softmax(-1)
    expected = torch.einsum("hqk,khd->qhd", probabilities, expanded_values)
    actual = flash_attn_with_kvcache(
        queries,
        pool.k_buffer[0].view(-1, page_size, kv_heads, head_dim),
        pool.v_buffer[0].view(-1, page_size, kv_heads, head_dim),
        page_table=(row[::page_size] // page_size).to(torch.int32)[None],
        cache_seqlens=torch.tensor(
            [retained + query_length], device=device, dtype=torch.int32
        ),
        cu_seqlens_q=torch.tensor([0, query_length], device=device, dtype=torch.int32),
        max_seqlen_q=query_length,
        causal=True,
        num_splits=1,
        ver=3,
    )
    error = (actual.float() - expected).abs().max().item()
    torch.testing.assert_close(actual.float(), expected, rtol=0.01, atol=0.005)
    return dict(
        length=length,
        page_size=page_size,
        window=window,
        query_length=query_length,
        head_dim=head_dim,
        max_absolute_error=error,
        **result,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("FA3 qualification requires a Hopper GPU")
    torch.manual_seed(20261007)
    results = [
        check(length, page, window, queries, dim)
        for length in (257, 1057)
        for page in (1, 64)
        for window in (None, 1, 8)
        for queries in (1, 5)
        for dim in (64, 128)
    ]
    report = {"passed": True, "gpu": torch.cuda.get_device_name(), "checks": results}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "checks": len(results),
                "max_absolute_error": max(r["max_absolute_error"] for r in results),
            }
        )
    )


if __name__ == "__main__":
    main()
