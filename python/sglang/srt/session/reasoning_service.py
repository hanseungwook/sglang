"""Transactional token-ID sessions with chronological replay outside GPU memory.

The backend implements open/acquire/park/close, prefill, generate and compact.
Transport loss does not cancel a transaction: retrying its request ID returns
the stored result. Explicit cancellation rolls back to the previous response.
"""

from __future__ import annotations

import asyncio
import copy
import fcntl
import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from sglang.srt.mem_cache.reasoning_eviction import (
    MessageBoundary,
    ReasoningEvictionPolicy,
    ReasoningEvictionReference,
    annotate_reasoning,
)


class SessionConflict(ValueError):
    status_code = 409


class ReplayRequired(ValueError):
    status_code = 503


def copy_record_for_append(record):
    """Isolate a transaction's mutable history without copying old responses.

    Committed response values, policy, identity and event entries are immutable
    inside the service. API responses are deep-copied on return. Token/message
    containers and the result dictionary are private to each attempt; events
    are replaced after execution, never mutated in place.
    """
    return {
        **record,
        "token_ids": list(record["token_ids"]),
        "messages": [dict(message) for message in record["messages"]],
        "results": dict(record["results"]),
    }


class ReasoningSessionService:
    def __init__(
        self,
        backend,
        *,
        markers,
        role_tokens,
        identity,
        context_length,
        vocab_size,
        outside_tokens=(),
        store=None,
    ):
        self.backend = backend
        self.markers = markers
        self.role_tokens = role_tokens
        self.identity = identity
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.outside_tokens = set(outside_tokens)
        self.store = Path(store) if store else None
        if self.store:
            self.store.mkdir(parents=True, exist_ok=True)
            self._store_lock = (self.store / ".lock").open("a")
            try:
                fcntl.flock(self._store_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                self._store_lock.close()
                raise ValueError(
                    "Reasoning session store is already in use by another server"
                ) from error
        self.records = {}
        self.resident = {}
        self.tasks = {}
        self.controls = set()

    def _path(self, session_id):
        if (
            not isinstance(session_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", session_id) is None
        ):
            raise ValueError("Invalid reasoning session ID")
        return self.store / f"{session_id}.json" if self.store else None

    def _save(self, record):
        path = self._path(record["session_id"])
        if path is None:
            return
        fd, temporary = tempfile.mkstemp(prefix=".reasoning-", dir=self.store)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(record, stream, separators=(",", ":"), allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(self.store, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _reference(self, record):
        messages = [MessageBoundary(**m) for m in record["messages"]]
        spans = annotate_reasoning(record["token_ids"], messages, self.markers)
        return ReasoningEvictionReference(
            record["token_ids"], spans, ReasoningEvictionPolicy(**record["policy"])
        )

    def _get(self, session_id):
        path = self._path(session_id)
        if session_id not in self.records:
            if path is None or not path.exists():
                raise ValueError("Unknown reasoning session")
            record = json.loads(path.read_text())
            if (
                record["schema_version"] != 1
                or record["session_id"] != session_id
                or record["identity"] != self.identity
            ):
                raise ValueError(
                    "Incompatible reasoning session model/tokenizer/template identity"
                )
            ref = self._reference(record)
            if record["events"] != [asdict(span) for span in ref.spans]:
                raise ValueError("Inconsistent reasoning session pruning events")
            self.records[session_id] = record
        return self.records[session_id]

    def open(self, window=None):
        policy = ReasoningEvictionPolicy(window)
        session_id = uuid.uuid4().hex
        record = dict(
            schema_version=1,
            session_id=session_id,
            identity=self.identity,
            policy=asdict(policy),
            token_ids=[],
            messages=[],
            events=[],
            results={},
        )
        self._save(record)
        self.records[session_id] = record
        return {
            "session_id": session_id,
            "policy": record["policy"],
            "persistent": self.store is not None,
        }

    async def _discard_backend(self, session_id):
        backend_id = self.resident.pop(session_id, None)
        if backend_id:
            await self.backend.close(backend_id)

    async def evict(self, session_id):
        self._get(session_id)
        if session_id in self.tasks or session_id in self.controls:
            raise SessionConflict("Cannot evict a session with an in-flight request")
        self.controls.add(session_id)
        try:
            await self._discard_backend(session_id)
        finally:
            self.controls.remove(session_id)
        return {"session_id": session_id, "replay_required": True}

    async def close(self, session_id):
        self._get(session_id)
        if session_id in self.tasks or session_id in self.controls:
            raise SessionConflict("Cancel or finish the current request before closing")
        self.controls.add(session_id)
        try:
            await self._discard_backend(session_id)
            path = self._path(session_id)
            if path is not None:
                path.unlink(missing_ok=True)
            self.records.pop(session_id)
        finally:
            self.controls.remove(session_id)
        return {"closed": True}

    async def cancel(self, session_id):
        self._get(session_id)
        entry = self.tasks.get(session_id)
        if entry is None:
            return {"cancelled": False}
        # Repeated cancel calls must not interrupt rollback or a control-reply
        # drain that is already handling the first cancellation.
        if not entry[2].cancelling():
            entry[2].cancel()
        try:
            await asyncio.shield(entry[2])
        except asyncio.CancelledError:
            pass
        return {"cancelled": True, "rolled_back": True}

    async def generate(
        self,
        session_id,
        *,
        request_id,
        input_ids,
        messages,
        sampling_params=None,
        return_logprob=False,
    ):
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError(
                "A nonempty request_id of at most 128 characters is required"
            )
        record = self._get(session_id)
        if session_id in self.controls:
            raise SessionConflict("Reasoning session is being closed or evicted")
        payload = dict(
            input_ids=input_ids,
            messages=messages,
            sampling_params={} if sampling_params is None else sampling_params,
            return_logprob=return_logprob,
        )
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        saved = record["results"].get(request_id)
        if saved:
            if saved["digest"] != digest:
                raise SessionConflict(
                    "request_id was already used with different input"
                )
            return copy.deepcopy(saved["response"])
        entry = self.tasks.get(session_id)
        if entry:
            if entry[:2] != (request_id, digest):
                raise SessionConflict(
                    "Reasoning session already has an in-flight request"
                )
            return copy.deepcopy(await asyncio.shield(entry[2]))
        # Validate before scheduling so rejected input never touches resident KV.
        candidate, params = self._prepare(record, **payload)
        task = asyncio.create_task(
            self._run(session_id, record, candidate, params, request_id, digest)
        )
        self.tasks[session_id] = (request_id, digest, task)

        def finished(done):
            if self.tasks.get(session_id, (None, None, None))[2] is done:
                del self.tasks[session_id]
            if not done.cancelled():
                done.exception()  # also retrieve errors after a transport disconnect

        task.add_done_callback(finished)
        return copy.deepcopy(await asyncio.shield(task))

    def _prepare(
        self, record, input_ids, messages, sampling_params, return_logprob=False
    ):
        if not isinstance(input_ids, list) or any(
            type(i) is not int or not 0 <= i < self.vocab_size for i in input_ids
        ):
            raise ValueError("input_ids must be a flat list of valid token IDs")
        if not isinstance(messages, list):
            raise ValueError(
                "messages must be a list of token-coordinate message boundaries"
            )
        if not isinstance(sampling_params, dict):
            raise ValueError("sampling_params must be an object")
        if type(return_logprob) is not bool:
            raise ValueError("return_logprob must be a boolean")
        allowed = {"max_new_tokens", "temperature", "top_p", "top_k", "sampling_seed"}
        if set(sampling_params) - allowed:
            raise ValueError(
                f"Unsupported reasoning sampling parameters: {sorted(set(sampling_params) - allowed)}"
            )
        params = {"temperature": 0.0, "max_new_tokens": 256, **sampling_params}
        maximum = params["max_new_tokens"]
        if type(maximum) is not int or maximum < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if len(record["token_ids"]) + len(input_ids) + maximum >= self.context_length:
            raise ValueError(
                "Request exceeds the model's original context-position limit"
            )
        candidate = copy_record_for_append(record)
        offset = len(record["token_ids"])
        continuing = bool(
            record["messages"] and not record["messages"][-1]["completed"]
        )
        if continuing:
            if input_ids or messages:
                raise ValueError(
                    "Continue an unfinished assistant with empty input_ids and messages"
                )
        else:
            if not messages:
                raise ValueError(
                    "Supply new messages ending in an open assistant message"
                )
            parsed = [MessageBoundary(**m) for m in messages]
            if parsed[-1].role != "assistant" or parsed[-1].completed:
                raise ValueError("The last new message must be an incomplete assistant")
            annotate_reasoning(input_ids, parsed, self.markers)
            previous = 0
            for message in parsed:
                if tuple(input_ids[message.start + 1 : message.body_start]) != tuple(
                    self.role_tokens[message.role]
                ):
                    raise ValueError("Message role/header does not match token IDs")
                if any(
                    token not in self.outside_tokens
                    for token in input_ids[previous : message.start]
                ):
                    raise ValueError("Unstructured tokens between messages")
                previous = message.end
                candidate["messages"].append(
                    asdict(
                        replace(
                            message,
                            start=message.start + offset,
                            body_start=message.body_start + offset,
                            end=message.end + offset,
                        )
                    )
                )
            candidate["token_ids"].extend(input_ids)
        params["stop_token_ids"] = [self.markers.message_end]
        params["skip_special_tokens"] = False
        params["_return_logprob"] = return_logprob
        return candidate, params

    async def _compact(self, backend_id, ref, boundary):
        before = ref.visible_positions(boundary - 1)
        plan = ref.plan_compaction(boundary, before, committed_through=boundary)
        return await self.backend.compact(
            backend_id,
            keep_indices=list(plan.keep_indices),
            expected_length=len(before),
            expected_offset=boundary - len(before),
        )

    async def _ensure_backend(self, session_id, record):
        backend_id = self.resident.get(session_id)
        if backend_id and (await self.backend.acquire(backend_id))["resident"]:
            return backend_id, False
        await self._discard_backend(session_id)
        backend_id = uuid.uuid4().hex
        self.resident[session_id] = backend_id
        await self.backend.open(backend_id)
        ref = self._reference(record)
        boundaries = {span.expires_at for span in ref.spans}
        for step in ref.replay_steps():
            await self.backend.prefill(
                backend_id, list(ref.token_ids[step.start : step.stop])
            )
            if step.stop in boundaries:
                await self._compact(backend_id, ref, step.stop)
        return backend_id, bool(record["token_ids"])

    async def _run(self, session_id, record, candidate, params, request_id, digest):
        try:
            # A cache loss between acquire and admission, or a decode retraction,
            # rolls back this attempt. Retry once from the last durable response.
            for attempt in range(2):
                try:
                    response, updated = await self._execute(
                        session_id, record, copy_record_for_append(candidate), params
                    )
                    break
                except ReplayRequired:
                    await self._discard_backend(session_id)
                    if attempt:
                        raise
            updated["results"][request_id] = {"digest": digest, "response": response}
            self._save(updated)
            self.records[session_id] = updated
            return response
        except BaseException:
            await self._discard_backend(session_id)
            raise

    async def _execute(self, session_id, record, candidate, params):
        backend_id, replayed = await self._ensure_backend(session_id, record)
        ref = self._reference(candidate)
        cursor = len(record["token_ids"])
        pages_freed = 0
        # New prompt history may itself contain completed assistant messages.
        for boundary in sorted(
            {s.expires_at for s in ref.spans if s.expires_at > cursor}
        ):
            await self.backend.prefill(backend_id, list(ref.token_ids[cursor:boundary]))
            pages_freed += (await self._compact(backend_id, ref, boundary))[
                "freed_pages"
            ]
            cursor = boundary
        result = await self.backend.generate(
            backend_id, list(ref.token_ids[cursor:]), params
        )
        output = result["output_ids"]
        finish = result["meta_info"]["finish_reason"]
        completed = bool(
            output
            and output[-1] == self.markers.message_end
            and finish["type"] == "stop"
        )
        candidate["token_ids"].extend(output)
        candidate["messages"][-1]["end"] = len(candidate["token_ids"])
        candidate["messages"][-1]["completed"] = completed
        ref = self._reference(candidate)
        if completed:
            # max_new_tokens=0 computes the previously sampled terminal token.
            await self.backend.prefill(backend_id, [])
            if any(s.expires_at == len(ref.token_ids) for s in ref.spans):
                pages_freed += (
                    await self._compact(backend_id, ref, len(ref.token_ids))
                )["freed_pages"]
        stats = await self.backend.park(backend_id)
        layout = ref.layout_after(len(ref.token_ids))
        candidate["events"] = [asdict(span) for span in ref.spans]
        result = copy.deepcopy(result)
        result["session_id"] = session_id
        result["meta_info"]["reasoning_eviction"] = {
            "window": ref.policy.window,
            "completed": completed,
            "completed_reasoning_tokens": sum(s.stop - s.start for s in ref.spans),
            "retained_tokens": len(layout.token_ids),
            "dropped_tokens": len(ref.token_ids) - len(layout.token_ids),
            "resident_kv_tokens": stats["committed_tokens"],
            "freed_pages": pages_freed,
            "next_absolute_position": layout.next_absolute_position,
            "replayed": replayed,
            "replay_tokens": len(record["token_ids"]) if replayed else 0,
        }
        return result, candidate
