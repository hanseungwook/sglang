"""Fail-closed configuration gate for the initial reasoning-eviction runtime."""


def validate_reasoning_eviction_config(cfg):
    if not cfg.enable_reasoning_kv_eviction:
        return
    required = {
        "enable_streaming_session": True,
        "disable_overlap_schedule": True,
        "tokenizer_worker_num": 1,
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "dcp_size": 1,
        "disaggregation_mode": "null",
        "disable_radix_cache": False,
        "device": "cuda",
    }
    for name, expected in required.items():
        if getattr(cfg, name) != expected:
            raise ValueError(f"Reasoning KV eviction requires {name}={expected!r}")
    backend = cfg.attention_backend
    if backend not in ("triton", "fa3"):
        raise ValueError("Reasoning KV eviction requires triton or fa3 attention")
    for name in (
        "enable_hierarchical_cache",
        "enable_lmcache",
        "enable_session_radix_cache",
        "enable_unified_cache_external_linker",
        "enable_page_major_kv_layout",
        "enable_lora",
        "skip_tokenizer_init",
        "allow_auto_truncate",
        "enable_mixed_chunk",
        "enable_two_batch_overlap",
        "enable_torch_compile",
        "enable_hisparse",
        "enable_pdmux",
        "enable_memory_saver",
        "enable_unified_memory",
        "strip_thinking_cache",
    ):
        if getattr(cfg, name, False):
            raise ValueError(f"Reasoning KV eviction does not support {name}")
    if cfg.speculative_algorithm is not None or cfg.quantization is not None:
        raise ValueError(
            "Reasoning KV eviction requires non-speculative, unquantized inference"
        )
    if cfg.dtype not in ("bfloat16", "bf16") or cfg.kv_cache_dtype not in (
        "auto",
        "bfloat16",
        "bf16",
    ):
        raise ValueError(
            "Reasoning KV eviction requires explicit BF16 model and KV dtype"
        )
    if cfg.model_impl not in ("auto", "sglang"):
        raise ValueError(
            "Reasoning KV eviction requires the native Horizon model implementation"
        )
    graphs = cfg.cuda_graph_config
    if graphs is None or any(
        getattr(graphs, phase).backend != "disabled" for phase in ("decode", "prefill")
    ):
        raise ValueError(
            "Reasoning KV eviction requires both CUDA graph backends disabled"
        )
    for name in ("prefill_attention_backend", "decode_attention_backend"):
        if getattr(cfg, name, None) not in (None, backend):
            raise ValueError(f"Reasoning KV eviction requires {name}={backend}")
