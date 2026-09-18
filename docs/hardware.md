# Tested host snapshot (2026-09-18)

| Component | Observed value | How established |
|---|---|---|
| GPUs | 2 × Intel Arc Pro B65 (`8086:e222`) | `lspci -nn` found two devices |
| VRAM | 32 GB GDDR6 **per card** | [Intel B65 specifications](https://www.intel.com/content/www/us/en/products/sku/245796/intel-arc-pro-b65-graphics/specifications.html), not a runtime allocation measurement |
| CPU | AMD Ryzen 5 7600X, 6 cores / 12 threads | `lscpu` |
| Host RAM | 32 GB installed (Linux reports about 30 GiB) | `free -h` |
| OS/kernel | Ubuntu 24.04.4 LTS, `7.0.0-31-generic` | `/etc/os-release`, `uname -r` |
| PCIe | x8/x8 reported by operator | Not re-verified by an independent link-state reading for this draft |
| Container budget | 16 GiB host RAM, no container swap | Docker configuration (`Memory=MemorySwap=17179869184`) |

The two 32 GB cards provide separate memory spaces. Tensor parallelism splits the model across GPUs and communicates over PCIe; do not infer that one request has a freely pooled 64 GB VRAM arena. The installed link width, Resizable BAR, power limits, firmware, and cooling may change results and should be recorded for independent reproductions.

Useful non-destructive checks:

```sh
lspci -nn | grep -Ei 'VGA|Display'
ls -l /dev/dri/renderD*
uname -r
docker info --format '{{.ServerVersion}}'
```

The observed PCI device ID maps to B65 in [Intel's Xe driver supported-hardware list](https://dgpu-docs.intel.com/overview/supported-hardware/xe-driver-gpus.html). Model support is not a promise that a particular kernel/userspace image combination is stable; always verify with an inference request.
