# From a 180K MTP-2 service to a 256K MTP-3 service on two Arc Pro B65s

*A follow-up to this cookbook's [September 17 comparison](../results/2026-09-17.md). Measurements and deployment state are as of September 23, 2026. This is one host's experiment log, not a claim that these settings are optimal for every B65 system.*

The short version: simply moving from our older vLLM build to the official 0.29.0 or 0.30.0 XPU images did not produce a convincing speedup. A locally built image combining **vLLM 0.30.0, PyTorch 2.13.0+xpu, and vllm-xpu-kernels 0.1.15** eventually passed a broader test at a 262,144-token context limit. On that image, MTP-3 generated tokens 7–19% faster than MTP-2 across eight synthetic single-request context lengths from 32K to 248K, at the cost of slightly longer time to first token (TTFT). We promoted that tested configuration and retained the previous service as an offline rollback. This does **not** establish a long-term production reliability record or answer-quality equivalence.

## The starting point and the upgrade path

The hardware and model did not change: two 32 GB Intel Arc Pro B65 cards, `Qwen/Qwen3.8-27B-FP8`, tensor parallelism 2, text-only serving, 16 GiB container RAM, no container swap, and no CPU weight offload. The previous service used vLLM `0.28.1rc1.dev681+ge7edf17ce.xpu`, PyTorch `2.13.0+xpu`, XPU kernels `0.1.14.1`, MTP-2, a 180,224-token context limit, `max-num-batched-tokens=256`, and `max-num-seqs=2`. The two GPUs have separate VRAM; TP=2 does not turn them into one pooled 64 GB card.

| When | Candidate or change | What we observed | Decision |
|---|---|---|---|
| Sep 17 | Official vLLM 0.29.0 XPU and a contemporary nightly, otherwise matched to the old service | Short single decode: 33.25 and 31.74 tok/s, respectively, versus a 33.11 tok/s interpolated control. Two cold ~125K prompts took about 407–408 s on both candidates. | Neither was a persuasive speed upgrade. |
| Sep 21 | Later XPU nightly, still PyTorch 2.13.0+xpu and kernels 0.1.14.1 | Batch 320 with two active sequences failed a concurrent-request gate; batch 2048 failed during startup. MTP-3 initially failed warm-up, then worked with one active sequence and batch 256. A later batch-512 warm-up failed. | Keep these as separate, conditional observations; do not promote the nightly. |
| Sep 21 | Experimental PyTorch 2.14.0+xpu + prebuilt kernels 0.1.15.3 | A standalone two-rank all-reduce passed, but vLLM TP=2 workers segfaulted in oneCCL/OFI before model weights loaded. | No tokens/s result; reject this stack. |
| Sep 22 | Official vLLM 0.30.0 XPU image, still kernels 0.1.14.1 | Matched short single decode was 37.87 versus 38.13 tok/s on the old service. Two ~125K concurrent prompts took 407.67 s, effectively the same as the earlier 0.29.0 result. | Do not promote the stock image for speed alone. |
| Sep 23 | vLLM 0.30.0 with a locally built kernels 0.1.15 wheel, initially using the old MTP-2/batch-256/seqs-2 profile | A matched `llama-benchy` matrix found small, mixed changes: short single generation +4.9%, short concurrent aggregate −1.5%, 32K prefill +0.7%, and 32K single generation +2.3%. | No clear speed case for replacing the old profile as-is. |
| Sep 23 | Same new image, batch 320, one active sequence, queue limit 3, context 262,144; first MTP-2, then MTP-3 | Both depths passed 15 main `llama-benchy` cases and two prefix-cache follow-ups. MTP-3 improved single-request generation at every tested context length. | Promote the MTP-3 profile after the complete retest. |

These rows are **different experiments**, not one ranking. In particular, the Sep 17/22 short-stream harness and the Sep 23 `llama-benchy` runs use different prompt construction and metric definitions; their tok/s numbers should not be subtracted from one another.

### What was actually installed

The official [vLLM 0.30.0 XPU image](https://github.com/vllm-project/vllm/releases/tag/v0.30.0) was pulled at manifest digest `sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d`. Its installed XPU-kernel package was **0.1.14.1**, despite the vLLM version being 0.30.0. We built a Linux x86-64 wheel from the official [vllm-xpu-kernels v0.1.15 tag](https://github.com/vllm-project/vllm-xpu-kernels/releases/tag/v0.1.15), source commit `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45`, and overlaid that wheel onto the 0.30.0 image without changing its PyTorch dependency. The resulting **local** image was verified to contain vLLM `0.30.0+xpu`, PyTorch `2.13.0+xpu`, Triton `3.7.2+xpu`, and XPU kernels `0.1.15`.

The local image ID is not a pullable registry digest, and this article does not claim that a public prebuilt image with this exact combination exists. Building kernels 0.1.15 was separate from the failed PyTorch 2.14 / kernels 0.1.15.3 experiment above.

## The decisive MTP-2 versus MTP-3 test

Both full runs used the *same locally built image*, model, TP=2, batch budget 320, one active sequence, queue limit 3, context limit 262,144, GPU-memory utilization 0.95, prefix-cache policy, and text-only mode. The intended variable was `num_speculative_tokens`: 2 versus 3. `llama-benchy 0.4.0` used synthetic prompts, `--exact-tg --no-cache`, and a 128-token generated output for each context-depth case. The 32K and 64K cases had two measured runs; each longer-context case had one. Cached-prefix follow-ups were measured separately and are **not** included in the uncached table.

| Context depth | MTP-2 generation | MTP-3 generation | Change | MTP-2 TTFT | MTP-3 TTFT |
|---:|---:|---:|---:|---:|---:|
| 32K | 35.0 tok/s | 37.9 tok/s | +8% | 36.7 s | 37.2 s |
| 64K | 31.9 | 37.7 | +18% | 85.4 s | 86.3 s |
| 96K | 28.4 | 33.5 | +18% | 146.4 s | 147.8 s |
| 128K | 29.3 | 31.8 | +9% | 219.7 s | 222.0 s |
| 176K | 27.7 | 29.5 | +7% | 354.4 s | 357.3 s |
| 224K | 25.3 | 29.1 | +15% | 520.3 s | 524.9 s |
| 240K | 23.4 | 27.8 | +19% | 582.6 s | 587.0 s |
| 248K | 24.4 | 29.1 | +19% | 614.8 s | 619.8 s |

The 248K case submitted about 254K actual prompt tokens after benchmark formatting, generated 128 tokens, and completed below the 262,144-token model limit. This is **not** a demonstration that a full 262,144-token input plus output fits. The MTP-3 run's full-suite counters accepted 4,002 of 6,225 drafted tokens (64.3%), versus 3,291 of 4,470 (73.6%) for MTP-2. A lower *fraction* is expected when another, harder draft position is added; what matters here is the measured generation speed, not acceptance percentage alone.

Both profiles completed all 17 scenarios: 15 main cases (short decode, 2K prefill, 32K–248K single-request depths, and queue cases up to four submitted requests) plus two prefix-cache checks. With `max-num-seqs=1`, the extra submitted requests waited rather than executing as two or four simultaneous model sequences. A cached 32K follow-up had TTFT about 1.8 s and a cached 64K follow-up about 2.9 s on MTP-3, but their *uncached* initial requests took about 37 and 86 s. Prefix-cache timings must not be presented as cold-document processing speed.

The long-context table is a small synthetic sample, especially above 64K. MTP-3's TTFT was consistently a little slower. The test checks successful generation and serving stability over this suite; it does not measure Goose task quality, token-by-token parity, sustained multi-day uptime, or two *executing* long requests on the promoted profile.

## Failed branches were part of the result

MTP-4 on the new image reached `/health` and passed a short API probe, but the first uncached 32K `llama-benchy` request triggered an Intel Xe `Engine memory CAT error`, compute-engine timeout, and reset. The failure reproduced on a second attempt. There is no valid MTP-4 speed or fourth-token-acceptance result from that run. The evidence implicates the MTP-4 execution path on this host and stack, but cannot isolate vLLM, the kernel package, the driver, or their interaction.

After those GPU resets, separate MTP-3/batch-4096 and MTP-3/batch-2048 candidates failed during profiling/warm-up before an API request could be sent. **Without rebooting the host**, we restarted the previously successful MTP-3/batch-320 configuration; it ran normally and passed the entire 17-case suite without a new GPU fault. The failed batch runs have no throughput results, and this sequence alone does not establish their root cause.

This is why a ready `/health`, a successful short prompt, or a large configured batch is not a sufficient promotion gate. We watched the kernel journal, container restarts, cgroup OOM/swap, and actual long requests, and kept an offline rollback.

## Promotion and what it means for an agent workload

After the full MTP-3 suite, the old service was renamed into a stopped rollback with restart disabled. A new production container was created from the **same image and command arguments** as the tested candidate, changing only the host port binding and restart policy. A configuration comparison verified the image, command, environment, mounts, memory/swap limits, IPC, security options, and devices before cutover. The candidate was stopped, then the new service started. The new service returned `/health` 200 and completed an authenticated short generation; it had zero restarts at that check. Its cgroup showed a 16 GiB memory limit and `memory.swap.max=0` / `memory.swap.current=0`. The rollback was not deleted.

The current serving profile is therefore: vLLM 0.30.0 + XPU kernels 0.1.15, Qwen3.8-27B-FP8, TP=2, MTP-3, `max-model-len=262144`, `max-num-batched-tokens=320`, `max-num-seqs=1`, `max-num-queued-reqs=3`, GPU-memory utilization 0.95, CPU offload 0, and text-only input. The exact client-side context number for 256 Ki tokens is **262144**, but an agent must leave room for generated output and further tool results.

An audit of recent Goose calls on the MacBook found many large inputs and a maximum completed input of 179,145 tokens in its 30-day window. That made the old 180,224-token limit uncomfortably close to observed usage. The audit did **not** record exact API start/end timestamps, so it could not determine how often two Goose calls truly overlap. Choosing one executing request plus a small waiting queue is a workload-driven preference, not proof that two executing requests would be slower for every user. A proper comparison needs request interval telemetry and matched long-context tests with `max-num-seqs=2`.

The promotion is a measured local engineering choice, not a general recommendation to install a mutable nightly or copy an unauthenticated production command. Anyone reproducing it should pin the base image digest, build and verify the exact kernel wheel, isolate the GPUs during testing, monitor Xe faults, and repeat the long-context and queue gates on their own host. See the [safe starting recipe](../recipes/README.md) for a deliberately conservative example; it has **not** been silently changed to this production profile.
