# PyTorch 2.15.0 for gfx906 on ROCm 10.0 (Debian package)

`python3-torch-gfx906` is unmodified upstream **PyTorch v2.15.0**, **torchvision 0.30.0** and **torchaudio 2.11.0**,
built by Exabit, Inc. against **ROCm 10.0.0-gfx906+20261018172427** (HIP 7.15.26333) for **gfx906** (Radeon VII,
Instinct MI50/MI60, Radeon Pro Vega II). Python 3.12, amd64. It installs to `/usr/lib/python3/dist-packages`, so
`import torch` works system-wide without a venv.

It replaces `python3-torch-gfx906` 2.14.1 (2026-10-22), built against the same ROCm build.

## Why this exists

AMD dropped gfx906 from ROCm, and with 2.15 PyTorch dropped the gfx906 kernels from its own ROCm wheels: its release
notes tell gfx906 users to stay on 2.14 or to build from source with gfx906 in `PYTORCH_ROCM_ARCH`. This package is that
source build. [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906) rebuilds ROCm from AMD's TheRock with gfx906
re-enabled and keeps PyTorch build recipes for it; this is a bare-metal build of the same stock sources, with Cascade
Lake CPU tuning and static oneMKL.

It uses no PyTorch patches.

## Install with apt

Use the rolling channel: one source line, and later versions arrive with `apt upgrade`.

```sh
sudo install -d -m 755 /etc/apt/keyrings
curl -fsSL https://exabit-io.github.io/resize-amdgpu-bars/keys/resize-amdgpu-bars.gpg \
    | sudo tee /etc/apt/keyrings/resize-amdgpu-bars.gpg > /dev/null
echo 'deb [signed-by=/etc/apt/keyrings/resize-amdgpu-bars.gpg] https://github.com/exabit-io/resize-amdgpu-bars/releases/download/python3-torch-gfx906 ./' \
    | sudo tee /etc/apt/sources.list.d/pytorch-gfx906.list
sudo apt update
sudo apt install python3-torch-gfx906
```

A machine that has the 2.14.1 or 2.13.0 source line (`.../releases/download/pytorch-2.14.1-gfx906-rocm10.0`) should
replace it with the line above; 2.14.1 and 2.13.0 are in the channel too, for going back. This release is also a signed
flat apt repository of its own (same key), for pinning exactly this version. The `.deb` carries a `debsigs` origin signature.

It needs a working **ROCm 10.0 gfx906** stack: the package depends on `amdrocm10.0-gfx906 (>= 10.0.0-gfx906+20261018172427)`
from the [gfx906 apt repository](https://github.com/mixa3607/ML-gfx906).

## Build configuration

| | |
|---|---|
| ROCm | 10.0.0-gfx906+20261018172427 (HIP 7.15.26333, RCCL 2.30.4) |
| GPU arch | `PYTORCH_ROCM_ARCH=gfx906` |
| CPU target | `-march=native -O3` -> **cascadelake**, AVX512F/DQ/BW/VL/CD + **AVX512-VNNI**, on every C/C++ compile (torch, torchvision, torchaudio) |
| fbgemm AVX512 kernels | `-mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512bf16 -mfma` on top, as in upstream's wheels; fbgemm picks its BF16 paths at run time |
| BLAS | Intel oneMKL 2026.2.1, statically linked |
| oneDNN | `USE_MKLDNN=ON` (v3.13) |
| Flash / mem-eff attention | OFF (the upstream gfx906 preset) |
| Distributed | RCCL from the ROCm stack (`USE_SYSTEM_NCCL`); new in 2.15: `libtorch_hip` links `libamd_smi` |
| Build | `python -m build --wheel --no-isolation` (scikit-build-core) |
| Build host | Xeon W-3275M (Cascade Lake-W), Mac Pro 7,1, 4x Vega 20 |

## What changes from 2.14.1 on this stack

- **A new dependency: `python3-typing-extensions (>= 4.13.0)`.** PyTorch 2.15 imports `typing_extensions.Format` and
  `get_annotations` whenever `torch` is imported. Ubuntu 24.04 ships typing_extensions 4.10.0, and with it `import torch`
  fails (`cannot import name 'Format'`). This release and the channel therefore also carry **`python3-typing-extensions`
  4.16.0-0exabit1**: the unmodified upstream wheel (sha256 `481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8`,
  checked against PyPI), under the same package name and file layout as Ubuntu's package, so apt upgrades it in place when
  it installs PyTorch. apt refuses the PyTorch package without it.
- **`torch.distributed`: `backend="nccl"` is now PyTorch's rewritten implementation** (opt-in in 2.14). On this stack it
  runs on RCCL 2.30.4, and `nccl`, `nccl-lazy` and `nccl-legacy` all give correct all-reduce, broadcast and all-gather
  results over four GPUs at the same speed (below). A process group is now bound to one device: pass
  `device_id=torch.device("cuda", local_rank)` to `init_process_group`. Upstream removes `nccl-legacy` and
  `TORCH_DIST_USE_NCCL2=0` in the next release.
- **`torch.topk` on the GPU is repeatable.** With tied values, 2.14.1 returns the tied indices in a different order on two
  identical calls (measured here on rows of 5,000 elements, for every k tried from 8 to 2,000); 2.15.0 returns the same
  order each time.
- `import torch` can no longer be executed twice in one process: `importlib.reload(torch)` raises.
- `libtorch_hip.so` now links `libamd_smi.so` (an upstream build dependency for ROCm 7.14 and later); the
  `amdrocm10.0-gfx906` dependency already brings it.
- torchvision 0.29.0 -> 0.30.0; torchaudio stays 2.11.0 (a newer commit of it).
- Everything else: [PyTorch 2.15.0 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.15.0).

## Verified on the build host

The installed package was run with `/usr/bin/python3`, no venv, on 4x gfx906 with GPUs capped at 125 W; the 2.14.1
package ran the same two scripts in the same session, just before the upgrade.

    torch 2.15.0+gfx906.20261018172427   hip 7.15.26333   cuda_available True   ndev 4
    MKL True   MKLDNN True   OpenMP True   CPU capability AVX512
    matmul 4096^3 per die: fp32 7.84-8.03 TF (err 2.2-2.8e-06), fp16 7.87-8.34 TF, bf16 4.56-4.88 TF
        2.14.1, same session:  fp32 7.86-8.06 TF,                 fp16 7.88-8.24 TF, bf16 4.54-4.86 TF
    MIOpen conv2d fwd/bwd OK; cross-device copy OK; P2P 0->1, 0->2, 2->3 True
    RCCL all_reduce over 4 GPUs: value correct, 1 GiB in 48 ms (21.00 GiB/s; 2.14.1: 20.95 GiB/s)
    torchvision 0.30.0: nms on the GPU equals the CPU result; roi_align GPU vs CPU max abs diff 0
    torchaudio 2.11.0: MelSpectrogram on the GPU

A second script compares the GPU with the CPU, one check per ROCm library the package links. **16 of 16 checks pass** on
2.15.0 (2.14.1: 15 of 16, the `topk` item above):

- rocBLAS gemm/gemv/bmm (fp32 relative error 9e-07, fp64 exact); solve, Cholesky, QR, singular values, eigenvalues and
  inverse in fp64 (worst relative error 4e-14); FFT (2e-07); sparse COO and CSR products and `coalesce`; random number
  streams (a seeded stream repeats)
- sort, stable sort, `topk` with and without ties, `argmax`/`argmin` over 2^25 elements, `unique`, `kthvalue`: equal to the CPU
- convolution with batch norm, pooling and transposed convolution, its input gradient, layer norm and group norm (4e-07);
  two-layer LSTM and GRU; scaled dot-product attention in fp32 (2e-07) and under autocast in fp16 and bf16
- a two-layer transformer trained for 30 AdamW steps reaches the same loss on the GPU and on the CPU (0.1241)
- scatter, `index_put` with accumulation, `EmbeddingBag`, masked selection; `cdist`/`pdist`; allocate and free 1 GiB;
  `torch.save` / `torch.load(weights_only=True)` across two devices
- `torch.distributed` over the four GPUs, all-reduce, broadcast and all-gather correct on each backend; 1 GiB all-reduce:
  `nccl` (the new implementation) 20.98 GiB/s, `nccl-lazy` 20.92, `nccl-legacy` 20.91 (2.14.1's `nccl`: 21.10)

Package checks: apt refuses the PyTorch deb when the newer `python3-typing-extensions` is not available; all 51 ROCm
libraries the binaries link belong to packages that the `amdrocm10.0-gfx906` dependency brings in; `-march=native` is
on 3,326 of 3,326 compile commands of the build.

## Licensing

PyTorch, torchvision and torchaudio are BSD-licensed. **`libtorch_cpu.so` statically links Intel oneMKL**, so this
package redistributes Intel binary code under the **Intel Simplified Software License**, which expressly permits
redistributing the binaries without modification. The full text and details are in
`/usr/share/doc/python3-torch-gfx906/`. Intel's OpenMP runtime is **not** included.

## Caveats

- Built and tested only on Cascade Lake + gfx906. With `-march=native` the CPU code **requires AVX512-VNNI** and will not run on hosts without it.
- No hipBLASLt gfx906 device library exists, so INT8 paths that need it still fall back; rocBLAS covers GEMM.
- There is no Triton in the package, so `torch.compile` with the Inductor GPU backend is not available; use eager mode.
- ROCm 10.0 is an unannounced stream in the gfx906 apt repository with no upstream counterpart; treat it as experimental.
- Installing this package upgrades Ubuntu's `python3-typing-extensions` from 4.10.0 to 4.16.0 for the whole system, not only for PyTorch.
- The wheel also declares `sympy>=1.13.3` and `setuptools>=77`; Ubuntu 24.04 has 1.12 and 68. As with 2.14.1, nothing tested here needs the newer ones, but `pip check` reports them.
- `torch.distributed`: with the new `nccl` backend a process group uses one GPU per process; several GPUs in one process need `nccl-legacy`, which upstream removes in the next release.
