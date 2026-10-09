# Experimental reasoning KV eviction

This opt-in runtime adds session-owned KV compaction, original RoPE positions,
and chronological recovery for multi-turn Horizon inference. The client sends
new token IDs and message boundaries; the server owns retention and replay.

**Validation status:** both 0.9B and 3.7B Horizon run with FA3 on H200 and pass
the session/compaction smoke checks below. The initial 0.9B Triton test passed
15 warm/cold greedy comparisons and restart recovery, but failed strict
sampled-logprob parity. With FA3, 0.9B cache rebuilding can also change greedy
reasoning text; a native control with eviction disabled reproduces that token
difference. This remains experimental. Full masked-reference model parity,
stress tests, and broad performance are not qualified. The optimization checks
below cover exact old/new compaction equivalence and scoped H200 timing.

## Policy

Prune after a **completed assistant message**, including its answer or tool
call. The assistant attends to all its current reasoning through the terminal
message token. The server forwards that terminal token before compacting KV.
The next user, tool, or assistant token sees the pruned history.

| Window | Retained from each completed reasoning block |
| --- | --- |
| `null` / Python `None` / CLI `full` | All tokens |
| `8` | Opening marker, last seven interior tokens, closing marker |
| `1` or `0` | Both markers, no interior tokens |
| Larger than the block | All tokens |

Every non-reasoning token is protected. All three IFM modes (`think`,
`think_fast`, `think_faster`) follow the same rule. Only leading assistant
reasoning is eligible, including prompt-supplied reasoning priming. Quoted
think tags in user/tool messages do not create spans. Missing, nested,
mismatched, or duplicate reasoning delimiters retain the affected block.
Ambiguous message delimiters reject the request rather than guessing roles.

Policy is fixed for a session. Open a new session to change it. Compaction
keeps existing KV values and their original positions. It does not extend the
model's original context-position limit or reduce the memory needed for the
current turn's full reasoning.

## Launch

Use this fork with dependencies matching `python/pyproject.toml`. The initial
configuration is dense, text-only `K2HorizonForCausalLM`, BF16 weights and KV,
one CUDA GPU, Triton or FA3 attention, eager execution, and one tokenizer worker.
FA3 is supported on Hopper (including H200). If prefill/decode backends are
specified separately, both must match the selected attention backend.
Pass a checkpoint compatible with the fork's native Horizon loader:

```bash
PYTHONPATH=python python -m sglang.launch_server \
  --model-path /path/to/dense-horizon-checkpoint \
  --model-impl sglang \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --enable-deterministic-inference \
  --enable-streaming-session \
  --enable-reasoning-kv-eviction \
  --disable-overlap-schedule \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --reasoning-session-store /path/to/reasoning-session-store
```

Use deterministic inference when investigating warm/cold parity. BF16
execution can produce different probabilities when replay prefills tokens
that were originally decoded one at a time. Initial 0.9B runs exceeded the
0.05 sampled-logprob tolerance even with no pruning, both with and without
deterministic inference; this flag alone does not guarantee replay parity.

For a Hugging Face checkpoint, use a pinned `--revision`. The store is optional;
without it, sessions survive resident KV loss but not a server restart. Each
store permits one server writer. Session files contain full transcripts and
cached responses; `close` deletes the corresponding file.

The feature defaults off. Unsupported configurations reject the request or
fail startup: quantization, speculative decoding, multiple GPUs/tokenizer
workers, disaggregated serving, hybrid attention, LoRA, CUDA graphs, overlap,
and external/hierarchical caches. The physical pool must be ordinary NHD MHA
BF16. Live weight updates are disabled while this feature is enabled; restart
with the new checkpoint and use new sessions.

## Multi-turn client

`client.py` provides `ReasoningSessionClient` and a two-turn example:

```bash
python examples/usage/reasoning_eviction/client.py \
  --tokenizer /path/to/dense-horizon-checkpoint --window 8 --force-replay
```

In a benchmark, open one session per conversation. Call `build_turn` with
**only new** user/tool/system messages, then pass its output to `generate`.
The helper renders IFM token IDs and an open assistant reasoning block.
It supports `reasoning_mode` and `reasoning_priming`. Use the same tokenizer
revision as the server. For a custom template, provide its exact token IDs
and trusted message boundaries directly; do not infer roles by scanning body
text or resend a shortened conversation.

The API returns a complete JSON response per request; token streaming is not
exposed in this first version. Native streaming sessions are used internally
to retain KV across requests.

| Endpoint (`POST`) | Body / behavior |
| --- | --- |
| `/reasoning_sessions/open` | `{"window": 8}`; returns a session ID and fixed policy |
| `/reasoning_sessions/{id}/generate` | `request_id`, `input_ids`, `messages`, optional `sampling_params` and `return_logprob` |
| `/reasoning_sessions/{id}/evict` | Drop resident KV, keeping the transcript for replay |
| `/reasoning_sessions/{id}/cancel` | Cancel the active transaction and roll back to the previous response |
| `/reasoning_sessions/{id}/close` | Release KV and remove the session, including persisted history |

Each message has `role`, `start`, `body_start`, `end`, and `completed`. Offsets
are relative to the submitted token-ID delta, with exclusive `end`.
`start` points at `<|ifm|im_start|>`; `body_start` follows the role tokens and
precedes the template newline. Completed messages include `<|ifm|im_end|>`.
The final new message must be an incomplete assistant prefix.

Choose a stable `request_id` **before** sending each turn. Retrying it with
identical arguments returns the stored response or waits for the same active
transaction; reusing it for different input returns HTTP 409. Only one new
transaction runs per session. A transport disconnect lets that transaction
finish so it can be retrieved by retry. Explicit cancellation discards the
attempt's KV and restores the last successful transcript.

If `meta_info.reasoning_eviction.completed` is false, the assistant was not
completed and its current reasoning remains intact. Continue with a new
request ID, `input_ids=[]`, and `messages=[]`. Finish that assistant before
adding another user/tool message. Completed turns and truncated continuations
are both recorded for recovery.

Responses include the usual output IDs/text and additional metrics:

- `completed_reasoning_tokens`: interior tokens in recognized completed spans.
- `retained_tokens` / `dropped_tokens`: cumulative logical residency counts.
- `resident_kv_tokens`: tokens whose KV is currently committed; a truncated
  response's last sampled token can still be uncommitted.
- `freed_pages`: physical pages freed by this transaction's compactions,
  excluding recovery work. A retained partial page stays allocated.
- `next_absolute_position`: position in the full, unshortened transcript.
- `replayed` / `replay_tokens`: whether prior history was reconstructed and
  how many transcript tokens were replayed.

## Recovery and ownership

Reasoning sessions neither read nor publish shared radix prefixes. The
coordinator retains full tokens, message boundaries, pruning events, policy,
model/tokenizer/template identity, and successful retry results outside GPU
memory. Idle KV may be evicted under pressure. Active transactions remain
pinned between their generation, terminal-forward, and compaction operations.
A decode retraction invalidates the attempt and triggers one replay retry.

Cold recovery teacher-forces the original transcript in chronological
segments, pruning at the same completed assistant boundaries. Prefilling only
retained tokens is incorrect: retained answer KV was originally computed
using reasoning that is no longer resident. Restart recovery checks identity
and saved events before replaying. Local checkpoint identity includes resolved
paths, sizes, and modification times; it is not a cryptographic weight digest.

Compaction finds the first removed token and copies only the retained suffix
after that hole. Earlier KV is already in its destination slots and is left
untouched. Each layer gathers the whole moving suffix before writing it, so
overlapping source/destination rows remain safe. Actual moves still synchronize
CUDA before copying and before freeing pages; a no-op avoids GPU work entirely.
Only whole unused pages are freed, keeping ownership of a partially filled final
page. New positions use each request's cumulative removed-token count; attention
and allocation still use resident lengths.

Transactions copy the token list, individual message boundaries, and the results
mapping. Previously committed responses and immutable identity/policy/event
values can be shared internally instead of recursively copying the entire retry
history twice per turn. Returned responses remain deep-copied. Failure or
cancellation discards the private candidate without changing committed history.

## Validation

Run the CPU suites with PyTorch installed; they load complete source modules
without importing the GPU server. Native session-hook tests use dependency
doubles for the scheduler and cache:

```bash
python test/registered/unit/mem_cache/test_reasoning_eviction.py -v
python test/registered/unit/mem_cache/test_reasoning_sessions.py -v
```

For randomized parity against the unmodified phase-2 training mask code,
also install NumPy and set `XLLM_REASONING_SOURCE=/path/to/xllm/data/reasoning.py`
when running the reference suite. A tiny two-layer rotary transformer compares
dense expiry-masked logits against compacted execution and replay, and verifies
that shortened-history recovery differs.

On a CUDA GPU, exercise the actual copy operations with real BF16 tensors and
test-double allocation records:

```bash
REASONING_KV_TEST_DEVICE=cuda \
  python test/registered/unit/mem_cache/test_reasoning_sessions.py PoolTests -v
```

For a running Horizon server, compare warm continuation with forced cold
replay across full/W0/W1/W8/oversized windows, including tool/user turns and
idempotent retries:

```bash
python examples/usage/reasoning_eviction/qualify_server.py \
  --tokenizer /path/to/dense-horizon-checkpoint \
  --output-dir /path/to/qualification-responses
```

This qualification requires identical greedy token IDs and compares sampled
token logprobs (default absolute tolerance 0.05 for BF16, configurable with
`--logprob-atol`). It reports per-turn compaction/replay metrics; the optional
output directory saves both responses before asserting parity so failures
can be inspected. Add `--keep-going` to collect failures across windows;
the command still exits unsuccessfully if any check fails. It does not
compare every vocabulary logit against a native Horizon masked-attention
oracle. Full model parity, numerical tolerance calibration, mixed-session
stress/allocator accounting, and latency/throughput remain release gates.

The 2026-10-07 real-model test used pretrained `IFM/K2-Horizon-0.9B` revision
`fa7f5ded0883803c8d8f2bdbfddcdba2e462e698`, one H200, BF16, Triton,
deterministic inference, eager execution, and an 8,192-token context limit.
Ordinary generation and full-history sessions matched tokens and sampled
logprobs exactly. All five windows passed three-turn greedy comparisons and
idempotent retries; W0 and W1 matched each other exactly. The first 137-token
transcript retained 35 tokens for W0/W1 and 42 for W8, preserving absolute
position 137. A process restart also restored the stored response exactly and
produced the same greedy continuation.

The maximum warm/cold sampled-logprob difference was 0.120 across windows and
0.081 after restart, exceeding the unchanged 0.05 threshold. The no-pruning
control's 0.113 difference reproduced exactly after flushing native SGLang's
cache, with the feature disabled; both native warm and native cold outputs
matched their corresponding full-history session outputs exactly. These
results establish functional smoke coverage, not strict numerical parity or
benchmark accuracy.

FA3 was additionally tested on H200 with the same 0.9B revision and 3.7B
revision `d6e5432c59637093c0e125e791d89ea4a0b53559` (the anti-doom-loop
release), at their native 131,072/524,288 context settings. Both passed native
versus session first-turn greedy agreement, real W1/W8 page reclamation,
tool-message continuation, and trimmed-search final-answer checks; concurrency
was four for 0.9B and one for 3.7B. These are short trajectories on servers
configured for the native limits, not full-length context qualification.

The FA3 attention check compares actual compacted BF16 KV against dense FP32
attention over the original, expiry-masked keys. It covers full/W1/W8, decode
and chunked prefill, nontrivial physical page mappings, page sizes 1/64, and
head dimensions 64/128. All 48 cases passed `atol=0.005, rtol=0.01`; the maximum
absolute error was 0.00435. Allocation records are test doubles. This does not
establish full-model masked-reference logit parity or benchmark quality.

The FA3 0.9B warm/cold qualification does not pass strict greedy or sampled-
logprob parity. A full-history example changed reasoning wording after rebuild
while preserving the final answer (45). An ordinary `/generate` control with
the eviction feature disabled reproduced the same warm/cold token sequences.
Native versus session cold logprobs matched exactly; their warm logprobs differed
by at most 0.0576, so exact numerical equivalence is not claimed. Preserve the
replay flag in evaluation traces when analyzing cache-loss effects.

The updated 3.7B FA3 checkpoint passed all 15 warm/cold comparisons across
full/W0/W1/W8/oversized windows, including idempotent retries, with identical
greedy token IDs and sampled logprobs. This paired test used four request rows
to keep both sessions resident; a single-row server necessarily evicts the
other session when alternating between the pair.

```bash
python examples/usage/reasoning_eviction/qualify_fa3_attention.py \
  --output /path/to/fa3-attention-results.json
```

## Measuring the compaction optimization

Compare against an unchanged checkout, not against a different retention window:
W8 and full history can produce different reasoning lengths and tool calls. The
scripts save individual timing samples and source hashes. Server comparisons also
save outputs, sampled logprobs, cache counts, and the exact launch commands.

```bash
# Run from the optimized checkout. Freeze the old function independently.
git show f5bb1c94c96cb3fa703f75849a1632a4e7ec8121:python/sglang/srt/session/reasoning_ops.py \
  > /tmp/baseline-reasoning-ops.py
python examples/usage/reasoning_eviction/benchmark_compaction.py \
  --baseline-ops /tmp/baseline-reasoning-ops.py --output /tmp/compaction.json
python examples/usage/reasoning_eviction/benchmark_metadata.py \
  --output /tmp/metadata.json
python examples/usage/reasoning_eviction/benchmark_server.py \
  --baseline-checkout /path/to/unchanged-checkout \
  --optimized-checkout "$PWD" --model /path/to/pinned-horizon-7b \
  --output-dir /tmp/server-comparison
# Independent real generated reasoning, interrupted continuations, and replay.
python examples/usage/reasoning_eviction/benchmark_server.py \
  --baseline-checkout /path/to/unchanged-checkout \
  --optimized-checkout "$PWD" --model /path/to/pinned-horizon-7b \
  --suite generation --output-dir /tmp/generated-comparison
```

On 2026-10-09, the compaction comparison used one H200, BF16, 36 layers,
eight KV heads, head dimension 128, and randomized physical page mappings.
Inputs and metadata were restored outside every measurement. After two warmup
pairs, nine paired samples counterbalanced baseline/optimized order. All 18
cases retained exactly the same K/V, token IDs, absolute positions and page-free
records. Allocation/request records are doubles, while the CUDA tensors and
production compaction functions are real.

For W8, 8,192 reasoning tokens and 128 answer tokens, page size 1:

| Unchanged prefix | Original median | Optimized median | Compaction speedup |
| --- | --- | --- | --- |
| 4,096 tokens | 2.756 ms | 1.406 ms | 1.96x |
| 65,536 tokens | 38.089 ms | 4.620 ms | 8.24x |
| 262,144 tokens | 151.708 ms | 14.991 ms | 10.12x |

In the largest case, copied KV fell from 262,281 tokens to 137 tokens and peak
temporary tensor allocation fell from approximately 514 MiB to 276 KiB. Page
size 64 gave similar results. These are **compaction timings, not generation
speedups**. The service still parses history, constructs retention plans, sends
control messages and forwards the terminal token; decode cost is unchanged for
an identical resident cache.

The server timing suite fixes the amount of work by supplying synthetic reasoning
and an answer prefix, then generating the message ending. It runs W8 with 4K/64K/
256K initial prefixes, 512/8K reasoning tokens per turn, and 4/16 turns. Both
implementations run sequentially on the same GPU with three repetitions. Median
timings include HTTP, metadata, prefill, generated ending, terminal forwarding,
and compaction. First repetitions can include shape-specific compilation; raw
samples must be inspected. This is not a long autoregressive reasoning throughput
measurement. The generation suite separately checks model-generated reasoning
and an interrupted continuation under full/W1/W8, with replay before turn three.

The 7B H200 serving comparison passed exact old/new token, sampled-logprob and
cache-statistic checks for every serial timed turn. Observed total trace medians
were:

| Initial prefix | Reasoning per turn | Turns | Original W8 | Optimized W8 | Time reduction |
| --- | --- | --- | --- | --- | --- |
| 4,096 | 512 | 4 | 0.528 s | 0.496 s | 6.1% |
| 65,536 | 512 | 16 | 9.338 s | 7.918 s | 15.2% |
| 65,536 | 8,192 | 16 | 23.403 s | 21.374 s | 8.7% |
| 262,144 | 8,192 | 4 | 54.181 s | 53.066 s | 2.1% |

The 256K initial prefill takes roughly 44 seconds in both implementations.
Excluding the first turn, its remaining three turns improved from 10.033 to
9.097 seconds (9.3%). For the 64K/512/16 case, the remaining 15 turns improved
from 5.230 to 3.878 seconds (25.8%). These figures compare old and optimized W8;
they do not compare W8 against full-history inference. Only three samples were
taken, and server process order was not randomized. They do not establish a
speedup for free-form BrowseComp trajectories or long autoregressive decoding.

Optimization regression tests include 144 repeated random compactions against
the full-gather oracle, unchanged-prefix/no-op access checks, and rollback of
failed completed and incomplete transactions. The 48 FA3 masked-attention checks
also pass after the change. Passing these checks does not establish full-model
masked-reference equivalence or benchmark accuracy.

The independent 7B generation suite passed on H200 with checkpoint revision
`3b6ae394c1b863c27a306d168b25555837955192`, BF16, FA3 and deterministic greedy
sampling. Old and optimized implementations produced identical token IDs,
sampled logprobs and cache statistics for three generated turns under each of
full/W1/W8, including a one-token interruption in every turn and forced replay
before turn three. Full/W0/W1/W8 supplied-reasoning replay controls also matched;
four concurrent sessions matched output tokens and cache statistics. Concurrent
logprob equality is not required because batch scheduling can differ. This is
equivalence between implementations using the same policy, not equality between
different windows or between warm and cold execution.

The standalone CPU trace evaluator remains available:

```bash
python examples/usage/reasoning_eviction/evaluate.py \
  test/registered/unit/mem_cache/fixtures/reasoning_eviction_training.json \
  --window 8 --chunk-tokens 16
```

Its synthetic training fixture has 78 tokens. W8 retains 71, drops seven, and
advances the next absolute position to 78. These are logical counts, not GPU
memory measurements. `ReasoningEvictionReference` exposes `attention_mask`,
`layout_after`, `plan_compaction`, and boundary-aware `replay_steps` for further
benchmark validation.
