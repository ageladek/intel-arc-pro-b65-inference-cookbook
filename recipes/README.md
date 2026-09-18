# vLLM XPU test profile

This Compose file is a portable **starting profile**, not a byte-for-byte copy of production. The image digest is the tested vLLM 0.29.0 XPU artifact. The 128K limit is deliberately below the observed long-context MTP failure zone. Production measurements in this cookbook used an older dev image and a 180,224-token limit; do not attribute those numbers to this exact Compose profile.

Prerequisites: Linux host with both B65 devices usable by the `xe` driver; Docker Engine and Compose; enough free disk for the model and image; no other process occupying the GPUs. Ensure `/dev/dri` exists and the container can access both render nodes. Do not assume `xpu-smi discovery` is reliable on every host: confirm device visibility with `lspci -nn`, `/dev/dri`, container startup logs, and a real inference request.

1. Copy `.env.example` to `.env`; set `HF_CACHE` to an absolute writable directory. `.env` is ignored by Git.
2. Inspect the rendered configuration: `docker compose --profile manual config`.
3. Stop other GPU inference services using your own service manager; the recipe does not do that for you.
4. Run `docker compose --profile manual up -d`, wait for `/health`, then run the synthetic benchmark.
5. Stop this test with `docker compose --profile manual down`; review GPU/kernel logs before returning another service to production.

`mem_limit: 16g` and `memswap_limit: 16g` constrain container *host* memory and disable container swap. They do not mean model weights are placed in RAM: `--cpu-offload-gb 0` requests no CPU weight offload. vLLM still needs some RAM for runtime, tokenization, and staging. The host's global swap configuration is independent.

The Compose port is bound to `127.0.0.1` on the host. There is no API key in this public example. Add authentication and a properly secured reverse proxy before remote access; never expose an unauthenticated vLLM endpoint to the internet.

Increasing `--max-model-len` to 180,224 should be treated as an experimental change with a separate long-context stability test. Do not combine it with a new image, batch size, MTP depth, or graph mode in one uncontrolled run.
