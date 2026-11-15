# PyTorch 2.15.0 for gfx906 on ROCm 10.1 (Debian package)

`python3-torch-gfx906` is unmodified upstream **PyTorch v2.15.0**, **torchvision 0.30.0** and **torchaudio 2.11.0**,
built by Exabit, Inc. against **ROCm 10.1.0-gfx906+20261018172427** (HIP 7.16.26385, LLVM 24) for **gfx906** (Radeon VII,
Instinct MI50/MI60, Radeon Pro Vega II). Python 3.12, amd64. It installs to `/usr/lib/python3/dist-packages`, so
`import torch` works system-wide without a venv.

This is the **ROCm 10.1 build** of the PyTorch 2.15.0 published earlier the same day against ROCm 10.0
(`2.15.0+gfx906.20261018172427-1`, release `pytorch-2.15.0-gfx906-rocm10.0`). Its version,
`2.15.0+rocm10.1.gfx906.20261018172427-1`, names the ROCm stream and sorts above the 10.0 build for apt and for pip, so it
is what the rolling channel installs. It also closes two dependency gaps of the 10.0 build (sympy, ninja; below).

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

The channel keeps the ROCm 10.0 builds (2.15.0, 2.14.1, 2.13.0) for going back. This release is also a signed flat apt
repository of its own (same key) with every package this one depends on, for pinning exactly this version. The `.deb`
carries a `debsigs` origin signature.

It needs **ROCm 10.1 for gfx906**: the package depends on `amdrocm10.1-gfx906 (>= 10.1.0-gfx906+20261018172427)` from the
[gfx906 apt repository](https://github.com/mixa3607/ML-gfx906). ROCm streams install side by side (`/opt/rocm/core-10.0`,
`/opt/rocm/core-10.1`): on a machine that has ROCm 10.0, apt adds the 10.1 packages when it upgrades this one, and removes
nothing. The libraries carry `RUNPATH=/opt/rocm/core-10.1/lib`, so they load ROCm 10.1 whatever `/opt/rocm` points at.

## Build configuration

| | |
|---|---|
| ROCm | 10.1.0-gfx906+20261018172427 (HIP 7.16.26385, LLVM 24, RCCL 2.30.7) |
| GPU arch | `PYTORCH_ROCM_ARCH=gfx906` |
| CPU target | `-march=native -O3` -> **cascadelake**, AVX512F/DQ/BW/VL/CD + **AVX512-VNNI**, on every C/C++ compile (torch, torchvision, torchaudio) |
| fbgemm AVX512 kernels | `-mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512bf16 -mfma` on top, as in upstream's wheels; fbgemm picks its BF16 paths at run time |
| BLAS | Intel oneMKL 2026.2.1, statically linked |
| oneDNN | `USE_MKLDNN=ON` (v3.13) |
| Flash / mem-eff attention | OFF (the upstream gfx906 preset) |
| Distributed | RCCL from the ROCm stack (`USE_SYSTEM_NCCL`); new in 2.15: `libtorch_hip` links `libamd_smi` |
| Build | `python -m build --wheel --no-isolation` (scikit-build-core) |
| Build host | Xeon W-3275M (Cascade Lake-W), Mac Pro 7,1, 4x Vega 20 |

## What changes from the ROCm 10.0 build

- **Built against ROCm 10.1** (HIP 7.16.26385, LLVM 24, RCCL 2.30.7) instead of 10.0. Same PyTorch source, same recipe,
  same speed (below). The package version now names the ROCm stream: `2.15.0+rocm10.1.gfx906.<build stamp>`.
- **`python3-sympy (>= 1.13.3)` is now a dependency, and the channel carries it.** With Ubuntu 24.04's sympy 1.12,
  `torch.export` with dynamic shapes, symbolic tracing (`make_fx(..., tracing_mode="symbolic")`) and `torch.compile` fail
  in PyTorch 2.15 with `ImportError: cannot import name '_args_sortkey' from 'sympy.core.basic'`. This release and the
  channel carry **`python3-sympy` 1.14.0-2~24.04.exabit1** (with `isympy-common` and `isympy3`): Ubuntu 26.04's source
  package, rebuilt for 24.04 in a clean chroot with no source change. With it all three work, including
  `torch.compile(dynamic=True)` on the CPU.
- **`ninja-build` and `g++` are now recommended packages.** `torch.utils.cpp_extension.load` needs ninja, and stopped with
  "Ninja is required to load C++ extensions" on a machine without it.
- `python3-typing-extensions (>= 4.13.0)` stays a dependency, served by the same channel (4.16.0-0exabit1), as in the
  ROCm 10.0 build.
- The ROCm 10.0 build of the same PyTorch stays in the channel (`2.15.0+gfx906.20261018172427-1`).

## Verified on the build host

The installed package was run with `/usr/bin/python3`, no venv, on 4x gfx906 with GPUs capped at 125 W. The reference is
the ROCm 10.0 build of the same PyTorch, run with the same scripts in the same session, just before the upgrade.

    torch 2.15.0+rocm10.1.gfx906.20261018172427   hip 7.16.26385   cuda_available True   ndev 4
    MKL True   MKLDNN True   OpenMP True   CPU capability AVX512
    matmul 4096^3 per die: fp32 7.84-8.03 TF (err 2.2-2.8e-06), fp16 7.91-8.28 TF, bf16 4.56-4.86 TF
        ROCm 10.0 build:   fp32 7.84-8.07 TF,                 fp16 7.92-8.30 TF, bf16 4.55-4.84 TF
    MIOpen conv2d fwd/bwd OK; cross-device copy OK; P2P 0->1, 0->2, 2->3 True
    RCCL all_reduce over 4 GPUs: value correct, 1 GiB in 47 ms (21.13 GiB/s; ROCm 10.0 build: 21.10 GiB/s)
    torchvision 0.30.0: nms on the GPU equals the CPU result; roi_align GPU vs CPU max abs diff 0
    torchaudio 2.11.0: MelSpectrogram on the GPU

- **GPU against CPU, one check per ROCm library** (`torch-ops-check.py`): **16 of 16 pass**, as on the ROCm 10.0 build
  (rocBLAS, rocSOLVER, rocFFT, rocSPARSE, rocRAND, MIOpen, sort/topk, attention in fp32 and under autocast, a short training
  run, indexing, `cdist`, memory and serialization).
- **`torch.distributed` over the four GPUs** on RCCL 2.30.7, 1 GiB all-reduce: `nccl` 21.36 GiB/s, `nccl-lazy` 21.30,
  `nccl-legacy` 21.12 (ROCm 10.0 build: 21.09 / 20.93 / 21.07); all-reduce, broadcast and all-gather correct on each.
- **The features that use sympy and a compiler** (`torch-pydeps-check.py`, CPU only): **6 of 6 pass**: `torch.export` with
  dynamic shapes, symbolic tracing, symbolic-shape simplification, `torch.compile(dynamic=True)` on the CPU, and a C++
  extension built through ninja and through setuptools. With Ubuntu's sympy 1.12 the first, second and fourth fail
  (`cannot import name '_args_sortkey' from 'sympy.core.basic'`); without ninja the ninja build fails.

Package checks: all 51 ROCm libraries the binaries link belong to ROCm 10.1 packages that the `amdrocm10.1-gfx906`
dependency brings in; `-march=native` is on 3,326 of 3,326 compile commands of the build.

## Licensing

PyTorch, torchvision and torchaudio are BSD-licensed. **`libtorch_cpu.so` statically links Intel oneMKL**, so this
package redistributes Intel binary code under the **Intel Simplified Software License**, which expressly permits
redistributing the binaries without modification. The full text and details are in
`/usr/share/doc/python3-torch-gfx906/`. Intel's OpenMP runtime is **not** included.

## Caveats

- Built and tested only on Cascade Lake + gfx906. With `-march=native` the CPU code **requires AVX512-VNNI** and will not run on hosts without it.
- No hipBLASLt gfx906 device library exists, so INT8 paths that need it still fall back; rocBLAS covers GEMM.
- There is no Triton in the package, so `torch.compile` with the Inductor GPU backend is not available; use eager mode.
- ROCm 10.1 is a stream of the gfx906 apt repository built from AMD's TheRock development line; treat it as experimental.
- ROCm 10.1 has no `rocm-smi`; `amd-smi` replaces it.
- Installing this package upgrades two Ubuntu packages for the whole system, not only for PyTorch: `python3-typing-extensions` 4.10.0 -> 4.16.0 and `python3-sympy` 1.12 -> 1.14.0 (with `isympy-common` and `isympy3` if they are installed).
- The wheel also declares `setuptools>=77.0.3`; Ubuntu 24.04 has 68.1.2, and this package does not replace it. The bound comes from PyTorch's own build metadata (setuptools 77 is the first with PEP 639 licence fields); at run time PyTorch only needs a setuptools that provides `build_ext`, and building a C++ extension through setuptools 68 is one of the checks above. `pip check` reports the unsatisfied bound.
- `torch.distributed`: with the new `nccl` backend a process group uses one GPU per process; several GPUs in one process need `nccl-legacy`, which upstream removes in the next release.
