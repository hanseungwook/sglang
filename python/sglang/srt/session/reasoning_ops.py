"""Scheduler-side operations for isolated reasoning sessions (eager, full KV)."""

from array import array

import torch


def has_reasoning_session_row(cache, req):
    session = getattr(req, "session", None)
    if not getattr(session, "reasoning_owned", False) or session.reasoning_cache_lost:
        return False
    slot = cache.session.slots.get(session.session_id)
    return slot is not None and slot.kv.holds_kv


def prefill_row_cost(cache, req):
    """Rows needed at admission; reasoning continuations borrow their own row."""
    if has_reasoning_session_row(cache, req) or req.kv.holds_kv:
        return 0
    beam = getattr(req, "beam_group", None)
    return beam.beam_width if beam is not None else 1


def apply_reasoning_positions(positions, reqs, extend_lens=None):
    """Derive offsets from the current request order, including filtered batches.

    Resident lengths still drive attention and allocation. Only RoPE positions
    advance over removed tokens. New tokens within each request are contiguous.
    """
    offsets = [getattr(req, "reasoning_position_offset", 0) for req in reqs]
    if not any(offsets):
        return positions
    if extend_lens is not None:
        offsets = [
            offset
            for offset, length in zip(offsets, extend_lens, strict=True)
            for _ in range(length)
        ]
    if len(offsets) != positions.numel():
        raise ValueError("Reasoning positions are not aligned with the forward batch")
    return positions + positions.new_tensor(offsets)


def compact_session(cache, session, *, keep_indices, expected_length, expected_offset):
    """Compact an idle, fully committed session; retain partial tail-page ownership.

    The coordinator has teacher-forced the terminal token in a prefill-only
    request. Only the suffix after the first hole moves; the preceding KV is
    already in its destination slots. Each layer gathers that entire suffix
    before writing, including when its source and destination overlap. CUDA
    synchronization deliberately trades boundary latency for stream safety.
    """
    if not session.reasoning_owned or session._inflight:
        raise ValueError("Compaction requires an idle reasoning-owned session")
    slot = cache.session.slots.get(session.session_id)
    if slot is None or session.reasoning_cache_lost:
        raise ValueError("Reasoning session KV was evicted; replay required")
    if session.reasoning_position_offset != expected_offset:
        raise ValueError("Stale reasoning compaction offset")
    kv = slot.kv
    if (
        kv.cache_protected_len
        or kv.cache_inserted_len
        or slot.lock is not None
        or kv.max_evicted_seqlen
        or kv.holds_mamba
    ):
        raise ValueError(
            "Reasoning compaction requires exclusively owned full-attention KV"
        )
    if kv.kv_committed_len != expected_length or kv.kv_allocated_len != expected_length:
        raise ValueError("Commit every token before reasoning compaction")
    if not keep_indices or keep_indices[-1] != expected_length - 1:
        raise ValueError(
            "Compaction must preserve ordered indices and the terminal token"
        )
    count = len(keep_indices)
    first_moved = count
    previous = -1
    for destination, source in enumerate(keep_indices):
        if type(source) is not int or source <= previous:
            raise ValueError(
                "Compaction must preserve ordered indices and the terminal token"
            )
        if first_moved == count and source != destination:
            first_moved = destination
        previous = source
    req = next(iter(session.req_nodes.values())).req
    if len(req.origin_input_ids) != expected_length or req.output_ids:
        raise ValueError("Compaction requires a completed prefill-only request")
    pool = cache.token_to_kv_pool_allocator.get_kvcache()
    # Quantized, page-major and hybrid pools need their own move/ownership rules.
    if (
        type(pool).__name__ != "MHATokenToKVPool"
        or pool.use_hnd
        or pool.kv_cache_layout != "nhd"
    ):
        raise ValueError("Reasoning compaction supports only the NHD MHA KV pool")
    if any(
        buf.dtype != torch.bfloat16 or buf.ndim != 3
        for buf in (*pool.k_buffer, *pool.v_buffer)
    ):
        raise ValueError("Reasoning compaction requires BF16 KV")
    if first_moved < count:
        row = cache.req_to_token_pool.req_to_token[kv.req_pool_idx]
        src = row[row.new_tensor(keep_indices[first_moved:], dtype=torch.long)].long()
        dst = row[first_moved:count].long()
        if src.is_cuda:
            torch.cuda.synchronize(src.device)
        for buffers in (pool.k_buffer, pool.v_buffer):
            for buf in buffers:
                buf[dst] = buf[src]  # advanced indexing materializes the RHS first
        if src.is_cuda:
            torch.cuda.synchronize(src.device)
    page_size = cache.page_size
    free_start = (count + page_size - 1) // page_size * page_size
    if free_start < expected_length:
        cache.free_kv_row(kv, [(free_start, expected_length)])
    kv.kv_allocated_len = kv.kv_committed_len = count
    retained = array("q", req.origin_input_ids[:first_moved])
    retained.extend(req.origin_input_ids[i] for i in keep_indices[first_moved:])
    req.origin_input_ids = retained
    req.origin_input_ids_unpadded = retained
    req.full_untruncated_fill_ids = array("q", retained)
    session.committed_origin_len = session.committed_unpadded_len = count
    session.committed_fill_len = count
    session.reasoning_position_offset += expected_length - count
    return {
        "resident_tokens": count,
        "dropped_tokens": expected_length - count,
        "freed_pages": (expected_length + page_size - 1) // page_size
        - (count + page_size - 1) // page_size,
        "next_absolute_position": count + session.reasoning_position_offset,
    }


def handle_reasoning_control(scheduler, request):
    from sglang.srt.managers.io_struct import (
        OpenSessionReqInput,
        ReasoningSessionControlOutput,
    )
    from sglang.srt.runtime_context import get_serving

    try:
        if not get_serving().enable_reasoning_kv_eviction:
            raise ValueError("Reasoning KV eviction is disabled")
        controller = scheduler.session_controller
        session = controller.get(request.session_id)
        if request.action == "open":
            if session is not None:
                raise ValueError("Backend session already exists")
            pool = scheduler.tree_cache.token_to_kv_pool_allocator.get_kvcache()
            if (
                type(pool).__name__ != "MHATokenToKVPool"
                or pool.use_hnd
                or pool.kv_cache_layout != "nhd"
            ):
                raise ValueError("Reasoning sessions require the NHD MHA KV pool")
            result = controller.open(
                OpenSessionReqInput(
                    capacity_of_str_len=0,
                    session_id=request.session_id,
                    streaming=True,
                )
            )
            if not result.success:
                raise ValueError("Cannot open reasoning backend session")
            session = controller.get(request.session_id)
            session.reasoning_owned = True
            session.reasoning_busy = True
            data = {"opened": True}
        elif request.action in ("acquire", "park"):
            if session is not None:
                if not session.reasoning_owned:
                    raise ValueError("Not a reasoning-owned session")
                session.reasoning_busy = request.action == "acquire"
            slot = scheduler.tree_cache.session.slots.get(request.session_id)
            data = {
                "resident": bool(session and slot and not session.reasoning_cache_lost),
                "committed_tokens": slot.kv.kv_committed_len if slot else 0,
            }
        elif request.action == "close":
            if session is not None:
                if not session.reasoning_owned:
                    raise ValueError("Not a reasoning-owned session")
                controller._close(request.session_id)
            data = {"closed": True}
        elif request.action == "compact":
            if session is None:
                raise ValueError("Missing backend session; replay required")
            data = compact_session(scheduler.tree_cache, session, **request.payload)
        else:
            raise ValueError("Unknown reasoning session operation")
        return ReasoningSessionControlOutput(success=True, data=data)
    except (ValueError, TypeError, torch.OutOfMemoryError) as error:
        return ReasoningSessionControlOutput(success=False, error=str(error))
