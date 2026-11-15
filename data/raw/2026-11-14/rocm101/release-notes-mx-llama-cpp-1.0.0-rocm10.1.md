# mx-llama.cpp for gfx906 on ROCm 10.1 (Debian package)

`mx-llama-cpp-gfx906` is [exabit-io/mx-llama.cpp](https://github.com/exabit-io/mx-llama.cpp) `master` at
[`df2c6e5c5`](https://github.com/exabit-io/mx-llama.cpp/commit/df2c6e5c5456c6a7b1bbcddfa703f59691c26ee9) (tag `gfx906/v1.0.0/r0/master`), built by Exabit, Inc. to run on
**ROCm 10.1.0-gfx906+20261018172427** (HIP 7.16.26385) for **gfx906** (Radeon VII, Instinct MI50/MI60, Radeon Pro Vega II).

This is the **ROCm 10.1 build** of the same source as the ROCm 10.0 build published earlier the same day
(`1.0.0+git20261114.df2c6e5c5.gfx906.20261018172427-1`, release `mx-llama-cpp-1.0.0-gfx906-rocm10.0`). Its version,
`1.0.0+git20261114.df2c6e5c5.rocm10.1.gfx906.20261018172427-1`, names the ROCm stream and sorts above the 10.0 build, so
it is what the rolling channel installs.

The GPU kernels are compiled with **ROCm 10.0's compiler** (AMD clang 23) and the package links and loads **ROCm 10.1's
libraries**. ROCm 10.1's own compiler (clang 24) produces a slower head-256 flash-attention kernel: prefill at one
255K-token stream dropped 4.1 % in our comparison, so that build was not published. The measurements are in the
section "ROCm 10.0 against ROCm 10.1" below.

It contains:

- [mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp), Marko Tombak's gfx906 fork of llama.cpp, up to `cdc81dae4` (the column-thread gated-delta-net kernel): q8 repack, custom all-reduce and tensor-parallel work, MMQ tile tables, DSpark/DFlash, Qwen3.8-Flash-Next and more.
- llama.cpp **v1.0.0** (b12359), merged into that fork, with two HIP fixes: the grouped top-k MoE fusion on 64-wide wavefronts, and `PAD_REFLECT_1D` launches of 2^32 or more threads. Offered to mxxm-t as [mxxm-t/mx-llama.cpp#24](https://github.com/mxxm-t/mx-llama.cpp/pull/24).
- **Exabit's gfx906 patches**: fused RMS norm + residual add + Q8_1 activations on the decode path, gated-delta-net producers folded into the GDN kernel, a 16-column MMVQ window, DPP warp reductions on GCN, an all-reduce size gate, and the max-ilp scheduler.

## Install with apt

Use the rolling channel: one source line, and later versions arrive with `apt upgrade`.

```sh
sudo install -d -m 755 /etc/apt/keyrings
curl -fsSL https://exabit-io.github.io/resize-amdgpu-bars/keys/resize-amdgpu-bars.gpg \
    | sudo tee /etc/apt/keyrings/resize-amdgpu-bars.gpg > /dev/null
echo 'deb [signed-by=/etc/apt/keyrings/resize-amdgpu-bars.gpg] https://github.com/exabit-io/resize-amdgpu-bars/releases/download/mx-llama-cpp-gfx906 ./' \
    | sudo tee /etc/apt/sources.list.d/mx-llama-cpp-gfx906.list
sudo apt update
sudo apt install mx-llama-cpp-gfx906
```

This release is also a signed flat apt repository of its own (same key), for pinning exactly this version. The `.deb` carries a `debsigs` origin signature.

It needs **ROCm 10.1 for gfx906** from the [gfx906 apt repository](https://github.com/mixa3607/ML-gfx906). The package
depends only on the runtime pieces it links: `amdrocm-runtime10.1`, plus the rocBLAS and RCCL host libraries and their
gfx906 kernel packages. ROCm streams install side by side (`/opt/rocm/core-10.0`, `/opt/rocm/core-10.1`): on a machine
that has ROCm 10.0, apt adds those 10.1 packages when it upgrades this one, and removes nothing.

## Build configuration

| | |
|---|---|
| ROCm at run time | 10.1.0-gfx906+20261018172427 (HIP 7.16.26385, RCCL 2.30.7) |
| GPU compiler | ROCm 10.0.0-gfx906+20261018172427's `clang++` (AMD clang 23.0.0git); recorded in `/usr/share/doc/mx-llama-cpp-gfx906/BUILD-INFO` |
| GPU arch | `AMDGPU_TARGETS=gfx906` |
| Defaults of the tree | `GGML_HIP_RCCL=ON`, `GGML_CUDA_FA_QUANTS=all` (every flash-attention K/V quant pair), `GGML_TP_AR_MAX_NE=20481`, `GGML_HIP_GFX906_MAX_ILP=ON` |
| CPU target | `-march=native` (C/C++ and `GGML_NATIVE`) for **Cascade Lake**: AVX512F/DQ/BW/VL/CD + AVX512-VNNI |
| Install prefix | `/opt/mx-llama.cpp` (`bin/`, `lib/`); every ELF carries `DT_RPATH=/opt/mx-llama.cpp/lib:/opt/rocm/core-10.1/lib` |
| Build host | Xeon W-3275M (Cascade Lake-W), Mac Pro 7,1, 4x Vega 20 |

Nothing is put on the default `PATH`, and no other llama.cpp/ggml installation can shadow the package's libraries:
`DT_RPATH` is searched before `LD_LIBRARY_PATH` and `ld.so.conf.d`. New in this build, the same holds for ROCm: the
binaries load ROCm 10.1 from `/opt/rocm/core-10.1/lib` whatever `/opt/rocm` points at (the 10.0 builds followed `/opt/rocm`,
so with both streams installed they ran on whichever one the alternatives selected). Run the tools from `/opt/mx-llama.cpp/bin`,
for example `/opt/mx-llama.cpp/bin/llama-server`.

## Verified on the build host

The installed package (`/opt/mx-llama.cpp`) was tested on 4x gfx906 (2x Radeon Pro Vega II Duo), with GPUs capped at 125 W and the generated RCCL topology in effect. The reference is the ROCm 10.0 build of the same commit, tested the same way earlier the same day:

- **Perplexity** (Qwen3.8-27B Q8_0, wikitext-2, 16K x 6 chunks, `-sm tensor`): **5.6093 +/- 0.0622**, identical to the ROCm 10.0 build.
- **`test-backend-ops`** (the package does not ship the tests; a source build of the same commit, compiled like the package by ROCm 10.0's compiler and run on ROCm 10.1): **20388/20388** on each of the four dies. The same count with ROCm 10.1's own compiler.
- **Output is byte-identical to the ROCm 10.0 build**: the greedy `-sm layer -lm dio` text and every MTP request (same SHA-256 of the text at `-np 1` and `-np 4`), with the same 154 of 201 drafts accepted.
- **MTP** (`llama-server --spec-type draft-mtp --spec-draft-n-max 2`, `-sm tensor`), 256 tokens in steady state: **73.1 / 73.2 / 73.5 tok/s** at `-np 1` (ROCm 10.0 build: 73.8 / 73.6 / 74.0) and 75.6 / 75.5 / 76.2 with `-np 4` and one active stream (10.0: 76.7 / 76.7 / 75.6). The first prompt (2,089 tokens) prefills at 723 tok/s (10.0: 743; one sample each). These are 8K smoke values; the comparison at the service's design points is the next section.
- **`-sm layer -lm dio`**: the model file is opened with `O_DIRECT`, no read falls back to buffered I/O.
- **Qwen3.8-Flash-Next** (qwen4exp, `-sm tensor --n-cpu-moe 41`): loads, and generation is coherent.
- **Decode-path folds** (source build as above): perplexity 5.0822 with the folds on (two runs) and 5.0575 with them off, as on ROCm 10.0.
- **Model fixtures** (source build as above): `test-llama-archs` (144 architectures) and `test-fp8` pass; `test-llama-context` behaves as on ROCm 10.0 (the KV-replica caveat below).
- **RCCL** 2.30.7 initialises and reads the generated topology (`/etc/rccl.conf`).
- **Build properties**: all 55 ELF files carry `DT_RPATH=/opt/mx-llama.cpp/lib:/opt/rocm/core-10.1/lib`. With `LD_LIBRARY_PATH` naming another llama.cpp build and ROCm 10.0's library directory, every ggml/llama library still comes from `/opt/mx-llama.cpp/lib` and all 26 ROCm libraries from `/opt/rocm/core-10.1`. `libggml-cpu` contains AVX-512 code (10,488 zmm and 1,189 `vpdpbusd` instructions).

## ROCm 10.0 against ROCm 10.1

Both comparisons use the same source commit and the service's two design points: Qwen3.8-27B Q8_0 on four dies (`-sm tensor`, f16 KV cache, 125 W per die, RCCL with the gated custom all-reduce), `llama-batched-bench` with 1,024 generated tokens, at 4 slots x 64K tokens and at 1 slot x 255K tokens. Five runs per arm per cell, the arms alternating in a seeded random order. The rule was written before the first run: a difference counts if it holds at q < 0.10 (exact permutation test, Benjamini-Hochberg over the four tests) and is 2 % or more.

**1. Compiled by and run on ROCm 10.1: rejected.**

| Cell | Metric | ROCm 10.0 | ROCm 10.1 | Difference | p / q |
|---|---|---|---|---|---|
| 4 x 64K | decode, tok/s per slot | 24.672 (24.34-24.83) | 24.691 (24.65-24.71) | +0.08 % | 0.93 / 0.93 |
| 4 x 64K | prefill, tok/s | 806.1 (804.8-807.5) | 794.8 (793.4-796.1) | -1.39 % | 0.008 / 0.032 |
| 1 x 255K | decode, tok/s | 31.959 (31.89-31.97) | 31.961 (31.91-32.01) | +0.01 % | 0.71 / 0.93 |
| 1 x 255K | prefill, tok/s | 410.2 (409.9-410.2) | 393.3 (391.8-393.9) | **-4.13 %** | 0.048 / 0.095 |

Values are the mean of the five runs at 4 x 64K and the median at 1 x 255K, with the range of the runs. Crossing the two (code from one compiler on the other's libraries, one run each) puts the whole loss on the compiler: 10.0-compiled code on the 10.1 libraries prefills at 410.8 and 805.1 tok/s, 10.1-compiled code on the 10.0 libraries at 392.2 and 795.3. Relinking the 10.1 build with single object files from the 10.0 compiler puts it in one of 215: the flash-attention tile kernels for head size 256. No compiler scheduling strategy or tile-table setting we tried restores it.

**2. This package (compiled by ROCm 10.0's compiler, running on ROCm 10.1): accepted.** The reference is a source build compiled by and run on ROCm 10.0.

| Cell | Metric | ROCm 10.0 source build | This package | Difference | p / q |
|---|---|---|---|---|---|
| 4 x 64K | decode, tok/s per slot | 24.787 (24.74-24.82) | 24.546 (24.54-24.55) | -0.97 % | 0.008 / 0.032 |
| 4 x 64K | prefill, tok/s | 806.0 (805.5-806.3) | 805.7 (805.5-806.0) | -0.03 % | 0.16 / 0.21 |
| 1 x 255K | decode, tok/s | 31.935 (31.90-31.97) | 31.698 (31.68-31.75) | -0.74 % | 0.048 / 0.095 |
| 1 x 255K | prefill, tok/s | 410.2 (410.1-410.5) | 410.2 (410.0-410.5) | -0.01 % | 1.0 / 1.0 |

Prefill is that of ROCm 10.0. Decode is 0.7 to 1.0 % lower than the source build on both cells: under the threshold, so the package is accepted, but consistent (every package run is below every reference run). A library-swap screen at 4 x 64K (three to four runs per arm, ranges not overlapping) splits it in two. About 0.6 % is the ROCm 10.1 runtime: the source build's own libraries give 24.81 tok/s per slot on ROCm 10.0 and 24.66 on ROCm 10.1. About 0.4 % is this package's host compiler flag: `-march=native` on `libllama` and `libllama-common` gives 24.56, the same two libraries without it 24.66 (the ROCm 10.0 packages carry the same flag; ggml's CPU backend is built for the native CPU either way). The 1 x 255K difference was not decomposed.

## Caveats

- Built and tested only on Cascade Lake + gfx906. `-march=native` means the CPU code **requires AVX-512 (with VNNI)** and will not run on hosts without it.
- ROCm 10.1 is a stream of the gfx906 apt repository built from AMD's TheRock development line; treat it as experimental.
- ROCm 10.1 has no `rocm-smi`; `amd-smi` replaces it (`amd-smi set -l HIGH`, `amd-smi set -o WATTS`).
- `GGML_HIP_RCCL=ON`: tensor split (`-sm tensor`) uses RCCL for prefill all-reduces. On the Mac Pro, RCCL needs the corrected XGMI topology (the firmware swaps the bridge links): about 25 GB/s all-reduce bus bandwidth without it, 34 with it; the `gfx906-rccl-topology` package generates it at boot.
- llama.cpp v1.0.0 removes API and options that were deprecated before (`--defrag-thold`, `llama_set_warmup` and others); scripts that still pass them stop with an unknown-argument error.
- With all four dies and a model whose layers share KV rows across devices (gemma4, mistral-large4), a sequence restored from a saved state can go wrong (the fork's KV replicas, in every release so far; `LLAMA_KV_REPLICA=0` avoids it). Qwen models are not affected.

## Licensing

llama.cpp, ggml and mx-llama.cpp are MIT-licensed; see `/usr/share/doc/mx-llama-cpp-gfx906/copyright`.
