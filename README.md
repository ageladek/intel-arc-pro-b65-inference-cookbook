# Dual Intel Arc Pro B65 inference cookbook

Reproducible notes for serving **Qwen3.8-27B-FP8** with vLLM XPU on two Intel Arc Pro B65 GPUs (tensor parallelism = 2). This is an independent, early-stage cookbook inspired by the evidence-first layout of [SergiioB's B70 cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook); no code or benchmark data is copied from it.

Status: **early public draft, updated 2026-09-23**. The example recipe is a *test profile*, not an instruction to replace a running production service. Never start it while another XPU model owns the GPUs.

**Standalone follow-up:** [From a 180K MTP-2 service to a 256K MTP-3 service](https://github.com/ageladek/intel-arc-pro-b65-vllm-upgrade-journey) covers the vLLM/XPU-kernel updates, failed configurations, full `llama-benchy` comparison, and production cutover. The recipe below remains the older conservative example, not the newly promoted profile.

## What is actually measured

| Setup, 2026-09-17 | Short single decode | Short concurrent-2 aggregate | Two cold ~125K prompts |
|---|---:|---:|---:|
| Local production, vLLM `0.28.1rc1.dev681`, MTP-2 | 33.11 tok/s (A1/A2 mean) | 63.63 tok/s (A1/A2 mean) | Not run in matched campaign |
| vLLM `0.29.0` XPU, MTP-2 | 33.25 tok/s | 66.25 tok/s | 407.28 s pair wall-clock |
| XPU nightly built 2026-09-17, MTP-2 | 31.74 tok/s | 64.15 tok/s | 408.21 s pair wall-clock |

The short gate used three single requests and one concurrent pair per candidate; differences of a few percent are not a reliable speedup. The long pair generated 256 tokens per request. The production A1/A2 mean is a control estimate, not an additional run. Definitions, test conditions, and limitations: [results/2026-09-17.md](results/2026-09-17.md).

## Tested platform

- 2 × Intel Arc Pro B65, 32 GB GDDR6 each, PCI device `8086:e222`; memory is **not** a single pooled 64 GB allocation. [Intel specifications](https://www.intel.com/content/www/us/en/products/sku/245796/intel-arc-pro-b65-graphics/specifications.html).
- AMD Ryzen 5 7600X, 32 GB system RAM, Ubuntu 24.04.4, Linux `7.0.0-31-generic` at the time of this snapshot.
- At the September 18 snapshot: vLLM XPU, Qwen3.8-27B-FP8, TP=2, MTP-2, `max-num-seqs=2`, 256 batched tokens, no CPU offload, no container swap. The then-production context limit was 180,224 tokens. The September 23 profile is described in the [standalone follow-up](https://github.com/ageladek/intel-arc-pro-b65-vllm-upgrade-journey).
- Production serves text only. Vision and document OCR are out of scope here.

The host facts are a point-in-time observation, not a required parts list. More detail: [docs/hardware.md](docs/hardware.md).

## Safe starting recipe

The [Compose profile](recipes/compose.yaml) pins the **tested vLLM 0.29.0 XPU image** by registry digest. It binds the API to host loopback, disables CPU offload and container swap, and deliberately does not configure an API key; do not expose it to a network without authentication.

```sh
cd recipes
cp .env.example .env
# Edit HF_CACHE in .env to an absolute local path containing the model cache.
docker compose --profile manual config
# Stop any other GPU model service yourself before this step.
docker compose --profile manual up -d
curl -f http://127.0.0.1:8000/health
docker compose --profile manual down
```

The commands above are **not run by this repository's checks**. The model may need to be downloaded on first start. See [recipes/README.md](recipes/README.md) before using the profile.

## Benchmark a local endpoint

The stdlib-only [benchmark harness](benchmarks/bench.py) sends synthetic prompts. It does not save response text or authorization headers.

```sh
python3 benchmarks/bench.py --url http://127.0.0.1:8000 --model Qwen3.8-27B-FP8 --single-runs 3 --pair-runs 1 --output-tokens 512
python3 -m unittest discover -s tests -v
```

For an authenticated endpoint, set `VLLM_API_KEY` in the shell; do not put it in Git. See [docs/lessons.md](docs/lessons.md) for interpretation, and [docs/publication-checklist.md](docs/publication-checklist.md) before any public push.

## Important limitations

- Speed is workload-, context-, image-, driver-, and metric-dependent. `decode tok/s` is not end-to-end tok/s, and aggregate concurrent throughput is not per-user speed.
- MTP-2 improved decode in our short depth test, but the former production build hit an EngineCore failure near its 180K context limit. Upstream [XPU kernel PR #600](https://github.com/vllm-project/vllm-xpu-kernels/pull/600) addresses the same assertion text; a later successful test does not by itself prove the exact root cause fixed or guarantee future stability.
- On September 23, a separate vLLM 0.30.0 + kernels 0.1.15 image passed a 256K/MTP-3 suite and was promoted; MTP-4 still faulted. See the [standalone upgrade report](https://github.com/ageladek/intel-arc-pro-b65-vllm-upgrade-journey) and the earlier [lessons](docs/lessons.md). This does not make the conservative Compose recipe a tested production replacement.

## Licenses

Code, tests, and configuration examples are licensed under the [MIT License](LICENSE-CODE). Written documentation and benchmark reports are licensed under [Creative Commons Attribution 4.0 International](LICENSE-DOCS.md). See those files for the exact scope and terms. Third-party software, models, and linked sources retain their own licenses.

## Sources

- [Intel Arc Pro B65 specifications](https://www.intel.com/content/www/us/en/products/sku/245796/intel-arc-pro-b65-graphics/specifications.html)
- [Qwen3.8-27B-FP8 model card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
- [vLLM 0.29.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)
- [vLLM XPU kernel releases](https://github.com/vllm-project/vllm-xpu-kernels/releases)
- Local test report: `results/2026-09-17.md` (sanitized, derived from private raw measurements)
