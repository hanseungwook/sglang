"""Tokenizer-manager adapter for the reasoning session coordinator."""

import asyncio
import hashlib
import json
import uuid
from pathlib import Path

from sglang.srt.managers.io_struct import GenerateReqInput, ReasoningSessionControlInput
from sglang.srt.mem_cache.reasoning_eviction import IFMReasoningMarkers
from sglang.srt.runtime_context import get_model, get_serving
from sglang.srt.session.reasoning_service import ReasoningSessionService, ReplayRequired


class TokenizerReasoningBackend:
    def __init__(self, manager):
        self.manager = manager

    async def control(self, action, session_id, **payload):
        self.manager.auto_create_handle_loop()
        # Drain a sent control operation even if explicit cancellation arrives;
        # otherwise its reply could satisfy the next queued operation's waiter.
        task = asyncio.create_task(
            self.manager.reasoning_session_communicator(
                ReasoningSessionControlInput(
                    action=action, session_id=session_id, payload=payload
                )
            )
        )
        try:
            responses = await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
        result = responses[0]
        if not result.success:
            if "replay required" in result.error.lower():
                raise ReplayRequired(result.error)
            raise ValueError(result.error)
        return result.data

    async def open(self, session_id):
        return await self.control("open", session_id)

    async def close(self, session_id):
        return await self.control("close", session_id)

    async def acquire(self, session_id):
        return await self.control("acquire", session_id)

    async def park(self, session_id):
        return await self.control("park", session_id)

    async def compact(self, session_id, **plan):
        return await self.control("compact", session_id, **plan)

    async def prefill(self, session_id, input_ids):
        result = await self.generate(session_id, input_ids, {"max_new_tokens": 0})
        if result["output_ids"]:
            raise ValueError("Prefill-only request unexpectedly generated tokens")

    async def generate(self, session_id, input_ids, sampling_params):
        rid = uuid.uuid4().hex
        params = dict(sampling_params)
        return_logprob = params.pop("_return_logprob", False)
        request = GenerateReqInput(
            input_ids=input_ids,
            rid=rid,
            stream=False,
            sampling_params=params,
            return_logprob=return_logprob,
            session_params={"id": session_id},
        )
        iterator = self.manager.generate_request(request, None)
        try:
            result = await iterator.__anext__()
            finish = result["meta_info"]["finish_reason"]
            if finish["type"] == "abort":
                raise ValueError(finish.get("message", "Reasoning generation aborted"))
            return result
        except ValueError as error:
            if "replay required" in str(error).lower():
                raise ReplayRequired(str(error)) from error
            raise
        except asyncio.CancelledError:
            self.manager.abort_request(rid)
            raise
        finally:
            await iterator.aclose()


def get_reasoning_service(manager):
    if not get_serving().enable_reasoning_kv_eviction:
        raise ValueError(
            "Launch with --enable-reasoning-kv-eviction to use reasoning sessions"
        )
    existing = getattr(manager, "reasoning_session_service", None)
    if existing is not None:
        return existing
    config = manager.model_config
    hf = config.hf_config
    if hf.architectures != ["K2HorizonForCausalLM"]:
        raise ValueError(
            "Reasoning KV eviction currently supports native K2HorizonForCausalLM only"
        )
    if config.is_multimodal or getattr(hf, "sliding_window", None):
        raise ValueError("Reasoning KV eviction requires text-only full attention")
    if any(getattr(hf, name, 0) for name in ("mova_num_experts", "num_experts")):
        raise ValueError("Reasoning KV eviction currently supports dense Horizon only")
    if config.quantization is not None:
        raise ValueError("Reasoning KV eviction requires an unquantized checkpoint")
    if str(config.dtype) != "torch.bfloat16":
        raise ValueError("Reasoning KV eviction requires BF16")
    tokenizer = manager.tokenizer
    markers = IFMReasoningMarkers.from_tokenizer(tokenizer)
    model = get_model()
    # HF commit identity covers remote artifacts. For local checkpoints also
    # record resolved paths and file size/mtime, detecting in-place replacement.
    model_path = Path(model.model_path)
    files = []
    if model_path.is_dir():
        for path in sorted(model_path.rglob("*")):
            if path.is_file() and path.suffix in (
                ".safetensors",
                ".bin",
                ".pt",
                ".json",
            ):
                stat = path.stat()
                files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    elif getattr(hf, "_commit_hash", None) is None:
        raise ValueError(
            "Recovery requires a resolved model revision or a local checkpoint"
        )
    identity = hashlib.sha256(
        json.dumps(
            {
                "implementation": 1,
                "model_path": str(model_path),
                "files": files,
                "model_revision": getattr(hf, "_commit_hash", None),
                "context_length": config.context_len,
                "dtype": str(config.dtype),
                "model_config": hf.to_dict(),
                "tokenizer": tokenizer.backend_tokenizer.to_str()
                if hasattr(tokenizer, "backend_tokenizer")
                else tokenizer.get_vocab(),
                "template": tokenizer.chat_template,
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    role_tokens = {
        role: tokenizer.encode(role, add_special_tokens=False)
        for role in ("system", "user", "assistant", "tool")
    }
    manager.reasoning_session_service = ReasoningSessionService(
        TokenizerReasoningBackend(manager),
        markers=markers,
        role_tokens=role_tokens,
        identity=identity,
        context_length=config.context_len,
        vocab_size=config.vocab_size,
        outside_tokens=[
            token
            for token in (
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                *markers.newline,
            )
            if token is not None
        ],
        store=get_serving().reasoning_session_store,
    )
    return manager.reasoning_session_service
