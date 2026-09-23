# What we learned on 2 × B65

This page records the earlier tests through September 18, 2026. The later vLLM 0.30.0 / XPU-kernels 0.1.15 tests and MTP-3 promotion are documented in the [upgrade journey](upgrade-journey-2026-09-23.md).

All statements below refer to Qwen3.8-27B-FP8 with vLLM XPU and TP=2 unless stated otherwise. They are measurements or observed failures, not general Intel GPU claims.

## MTP depth

A short matched-prompt, `temperature=0` depth test on vLLM 0.29.0 (batch budget 256, XPU Graph off) measured:

| Draft depth | Single decode | Median TTFT | MTP accepted / drafted |
|---|---:|---:|---:|
| MTP off | 17.28 tok/s | 0.131 s | — |
| MTP-1 | 28.97 tok/s | 0.192 s | 1063/1337 = 79.5% |
| MTP-2 | 34.68 tok/s | 0.200 s | 1387/2080 = 66.7% |

The MTP-2 *rate* is lower than MTP-1's, but it proposes more tokens per step and was faster in this short test. Acceptance is workload-specific. MTP-4 caused `DEVICE_LOST` in startup warm-up on this setup; there is no valid throughput number for it. These speed checks do **not** prove reasoning quality or token/logit equivalence.

## Long-context stability

Production with MTP-2 and a 180,224-token context limit has hit an XPU GDN assertion near the limit:

```text
Expected spec_token == num_spec_decodes * (num_speculative_tokens + 1)
```

The fatal EngineCore error restarted the serving container. Upstream [vllm-xpu-kernels PR #600](https://github.com/vllm-project/vllm-xpu-kernels/pull/600) fixes ragged speculative-token traversal and replaces that exact assertion in the kernel source. A 2026-09-18 nightly still carried the same `vllm_xpu_kernels 0.1.14.1` binary as the 2026-09-17 test image and still contained the old assertion, so **a changed nightly tag did not mean the fix was included**. Verify package provenance and the actual binary before claiming the issue fixed; then stress-test long and concurrent requests.

For an unattended agent, a lower client auto-compaction threshold and a shorter server context may reduce exposure, but neither is a kernel fix. A request can grow substantially within one agent turn.

## Scheduler and graphs

- `max-num-batched-tokens=256` is the production baseline. vLLM warns it may underfill MTP slots. We tried larger budgets, but did not establish a repeatable promotion-worthy improvement; 384 and deeper MTP encountered GPU failures in separate tests. Change one variable at a time.
- XPU Graph with MTP-2, conservative capture sizes, and TP=2 did not improve a matched ~11K-input test: TTFT 11.83 s with graph versus 11.77 s without; decode 31.13 versus 33.10 tok/s. This is one test, not a universal graph verdict.
- QK Norm + RoPE fusion (PR #49394) gave only about 1–1.6% in a short test, below our 5% screening threshold. Revisit only as part of a larger, otherwise worthwhile image update.
- Cold concurrent long prefill remains costly: two ~125K requests took roughly 407–408 seconds wall-clock in both tested candidates. Do not present the concurrent aggregate number as per-user generation speed.

## Promotion gate

Use A1 → candidate → A2 on the same hardware. Hold model, tokenizer, quantization, context, batch, MTP depth, prefix-cache policy, prompts, output length, and metric definitions constant. Include short decode, 32K and 125K concurrent prefill, deterministic answer/marker checks, at least a mixed 10-minute soak, MTP acceptance, host RAM, swap, `dmesg` Xe faults, and EngineCore errors. Promote only after a longer independent run confirms both speed and stability.

Do not run a test service beside production on the same two GPUs. Do not assume a successful HTTP 200 proves output parity or that an XPU kernel fix is present.
