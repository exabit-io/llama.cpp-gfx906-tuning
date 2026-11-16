# mx-llama.cpp for gfx906 on ROCm 10.1, compiled by ROCm 10.1 (Debian package)

`mx-llama-cpp-gfx906` is [exabit-io/mx-llama.cpp](https://github.com/exabit-io/mx-llama.cpp) `master` at
[`290fc642a`](https://github.com/exabit-io/mx-llama.cpp/commit/290fc642ac23fc8ec73c745f40cd315c7ff32d8c) (tag `gfx906/v1.0.0/r1/master`), built by Exabit, Inc. to run on
**ROCm 10.1.0-gfx906+20261018172427** (HIP 7.16.26385) for **gfx906** (Radeon VII, Instinct MI50/MI60, Radeon Pro Vega II).

Version `1.0.0+git20261115.290fc642a.rocm10.1.gfx906.20261018172427-1` replaces
`1.0.0+git20261114.df2c6e5c5.rocm10.1.gfx906.20261018172427-1` as the newest version in the rolling channel. It is the
same llama.cpp (v1.0.0) and the same patches. Two things changed in how it is built:

1. **It is compiled by ROCm 10.1's own compiler** (AMD clang 24). The previous build used ROCm 10.0's compiler, because
   clang 24 produced a slower head-256 flash-attention kernel (prefill -4.1 % at one 255K-token stream). The cause is
   found and the source tree now carries a one-object compiler option that removes it (the one commit this build adds;
   section "The compiler" below). Building the package no longer needs the ROCm 10.0 SDK.
2. **The host code is no longer compiled with a global `-march=native`.** On `libllama` and `libllama-common` that flag
   cost 0.4 % of decode at 4 slots x 64K. ggml's CPU backend is still built for the native CPU (`GGML_NATIVE`), so the
   AVX-512 requirement is unchanged.

It contains:

- [mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp), Marko Tombak's gfx906 fork of llama.cpp, up to `cdc81dae4` (the column-thread gated-delta-net kernel): q8 repack, custom all-reduce and tensor-parallel work, MMQ tile tables, DSpark/DFlash, Qwen3.8-Flash-Next and more.
- llama.cpp **v1.0.0** (b12359), merged into that fork, with two HIP fixes: the grouped top-k MoE fusion on 64-wide wavefronts, and `PAD_REFLECT_1D` launches of 2^32 or more threads. Offered to mxxm-t as [mxxm-t/mx-llama.cpp#24](https://github.com/mxxm-t/mx-llama.cpp/pull/24).
- **Exabit's gfx906 patches**: fused RMS norm + residual add + Q8_1 activations on the decode path, gated-delta-net producers folded into the GDN kernel, a 16-column MMVQ window, DPP warp reductions on GCN, an all-reduce size gate, the max-ilp scheduler, and (new) the LLVM 24 rule for the head-256 tile kernels.

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
| GPU compiler | ROCm 10.1.0-gfx906+20261018172427's `clang++` (AMD clang 24.0.0git); recorded in `/usr/share/doc/mx-llama-cpp-gfx906/BUILD-INFO` |
| GPU arch | `AMDGPU_TARGETS=gfx906` |
| Defaults of the tree | `GGML_HIP_RCCL=ON`, `GGML_CUDA_FA_QUANTS=all` (every flash-attention K/V quant pair), `GGML_TP_AR_MAX_NE=20481`, `GGML_HIP_GFX906_MAX_ILP=ON` (with clang 24 or newer this also sets `-mllvm -amdgpu-vgpr-threshold-percent=90` on the head-256 tile instance) |
| CPU target | CMake Release defaults; ggml's CPU backend with `GGML_NATIVE` for **Cascade Lake**: AVX512F/DQ/BW/VL/CD + AVX512-VNNI |
| Install prefix | `/opt/mx-llama.cpp` (`bin/`, `lib/`); every ELF carries `DT_RPATH=/opt/mx-llama.cpp/lib:/opt/rocm/core-10.1/lib` |
| Build host | Xeon W-3275M (Cascade Lake-W), Mac Pro 7,1, 4x Vega 20 |

Nothing is put on the default `PATH`, and no other llama.cpp/ggml installation can shadow the package's libraries:
`DT_RPATH` is searched before `LD_LIBRARY_PATH` and `ld.so.conf.d`. The same holds for ROCm: the binaries load ROCm 10.1
from `/opt/rocm/core-10.1/lib` whatever `/opt/rocm` points at. Run the tools from `/opt/mx-llama.cpp/bin`, for example
`/opt/mx-llama.cpp/bin/llama-server`.

## Verified on the build host

The installed package (`/opt/mx-llama.cpp`) was tested on 4x gfx906 (2x Radeon Pro Vega II Duo), with GPUs capped at 125 W and the generated RCCL topology in effect. The reference is the previous ROCm 10.1 package (compiled by ROCm 10.0's compiler), tested the same way the day before:

- **Perplexity** (Qwen3.8-27B Q8_0, wikitext-2, 16K x 6 chunks, `-sm tensor`): **5.6093 +/- 0.0622**, identical to the previous package.
- **`test-backend-ops`** (the package does not ship the tests; a source build of the same commit by the same compiler, GPU code byte-identical to the package's in all 215 objects): **20388/20388** on each of the four dies.
- **Output is byte-identical to the previous package**: the greedy `-sm layer -lm dio` text and every MTP request (same SHA-256 of the text at `-np 1` and `-np 4`), with the same 154 of 201 drafts accepted.
- **MTP** (`llama-server --spec-type draft-mtp --spec-draft-n-max 2`, `-sm tensor`), 256 tokens in steady state: **73.5 / 73.4 / 73.7 tok/s** at `-np 1` (previous package: 73.1 / 73.2 / 73.5) and 76.1 / 76.1 / 76.2 with `-np 4` and one active stream (previous: 75.6 / 75.5 / 76.2). The first prompt (2,089 tokens) prefills at 749 tok/s (previous: 723; one sample each). These are 8K smoke values; the comparison at the service's design points is the next section.
- **`-sm layer -lm dio`**: the model file is opened with `O_DIRECT`, no read falls back to buffered I/O.
- **Qwen3.8-Flash-Next** (qwen4exp, `-sm tensor --n-cpu-moe 41`): loads, and generation is coherent.
- **Decode-path folds** (source build as above): perplexity 5.0822 with the folds on (two runs) and 5.0575 with them off, as with every build of this source.
- **Model fixtures** (source build as above): `test-llama-archs` (144 architectures) and `test-fp8` pass; `test-llama-context` behaves as before (the KV-replica caveat below).
- **RCCL** 2.30.7 initialises and reads the generated topology (`/etc/rccl.conf`).
- **Build properties**: all 55 ELF files carry `DT_RPATH=/opt/mx-llama.cpp/lib:/opt/rocm/core-10.1/lib`. With `LD_LIBRARY_PATH` naming another llama.cpp build and ROCm 10.0's library directory, every ggml/llama library still comes from `/opt/mx-llama.cpp/lib` and all 26 ROCm libraries from `/opt/rocm/core-10.1`. `libggml-cpu` contains AVX-512 code (10,488 zmm and 1,189 `vpdpbusd` instructions).

## The compiler

All measurements: Qwen3.8-27B Q8_0 on four dies (`-sm tensor`, f16 KV cache, 125 W per die, RCCL with the gated custom all-reduce), `llama-batched-bench`.

**What LLVM 24 did.** One of the 215 GPU objects carried the whole loss: the flash-attention tile kernels for head size 256, and among them the one prefill runs (16 query columns x 2 heads per block). Compiled by clang 24 with the tree's `max-ilp` scheduler setting, its inner loop reloads 10 spilled vector registers from the stack in every iteration (clang 23: none). Running each compiler's optimized bitcode through the other compiler's code generator places it: clang 23's bitcode through LLVM 24's code generator is slow, clang 24's bitcode through LLVM 23's is fast. The driver's new offload pipeline (device code through LTO) is not involved; either compiler is as fast or as slow with it as without.

**The rule.** LLVM 24 has a new switch that scales the vector-register limit its scheduler works to. At 90 % (`-mllvm -amdgpu-vgpr-threshold-percent=90`) the kernel has no spills. The tree applies it to that one source file when the compiler is clang 24 or newer; 21 of the object's 77 kernels change, the decode kernel does not. One other switch also removes the spills (`-amdgpu-use-amdgpu-trackers=1`) but costs 5 % of decode at 128K depth on both compilers and is not used.

Three builds that differ only in the GPU library, on the ROCm 10.1 runtime, three runs each in shuffled order (mean and range):

| Cell | Metric | clang 23 (ROCm 10.0) | clang 24 (ROCm 10.1) | clang 24 with the rule |
|---|---|---|---|---|
| 4 x 64K | decode, tok/s per slot | 24.706 (24.697-24.715) | 24.715 (24.697-24.725) | 24.695 (24.687-24.707) |
| 4 x 64K | prefill, tok/s | 808.2 (808.1-808.3) | 797.4 (797.3-797.5) | 809.5 (809.4-809.6) |
| 1 x 128K | decode, tok/s | 40.036 (40.000-40.063) | 40.121 (40.013-40.195) | 40.109 (40.088-40.132) |
| 1 x 128K | prefill, tok/s | 605.2 (604.9-605.4) | 590.0 (589.2-590.8) | 606.8 (606.6-606.9) |

**Acceptance of this package** at the service's two design points, 1,024 generated tokens, against a source build of `df2c6e5c5` compiled by and run on ROCm 10.0. Five runs per arm per cell, the arms alternating in a seeded random order. The rule was written before the first run: a difference counts if it holds at q < 0.10 (exact permutation test, Benjamini-Hochberg over the four tests) and is 2 % or more.

| Cell | Metric | ROCm 10.0 source build | This package | Difference | p / q |
|---|---|---|---|---|---|
| 4 x 64K | decode, tok/s per slot | 24.868 (24.84-24.89) | 24.737 (24.71-24.76) | -0.52 % | 0.008 / 0.016 |
| 4 x 64K | prefill, tok/s | 808.1 (806.2-809.1) | 810.3 (809.2-811.0) | +0.27 % | 0.008 / 0.016 |
| 1 x 255K | decode, tok/s | 32.034 (31.93-32.07) | 32.051 (32.02-32.06) | +0.05 % | 0.57 / 0.57 |
| 1 x 255K | prefill, tok/s | 411.3 (411.1-411.7) | 412.4 (412.2-412.7) | +0.28 % | 0.048 / 0.063 |

Values are the mean of the five runs at 4 x 64K and the median at 1 x 255K, with the range of the runs. No difference reaches 2 %, so neither cell counts as changed and the package is accepted. Prefill is 0.3 % above the ROCm 10.0 build in both cells, every run above every reference run. Decode at 4 x 64K is 0.5 % below it, every run below every reference run: 0.3 % of that is the runtime (next paragraph), the rest was not decomposed. Decode at 1 x 255K does not separate. The previous package, in its own acceptance run against the same reference build, read -0.97 % and -0.74 % decode.

**The ROCm 10.1 runtime's decode cost** (0.3 % at 4 x 64K against ROCm 10.0's libraries, same code) is in ROCr, the HSA runtime. ROCm 10.0's ROCr polls its asynchronous events in a loop that never sleeps and keeps one host core busy for the whole run; ROCm 10.1's naps between polls. The package changes nothing there: it is ROCm's trade of a host core for 0.3 % of decode.

**Other compiler options.** A survey of 21 single changes to how the GPU code is compiled (optimization level, vectorizers, unrolling, inlining, scheduler strategies, target ID, register and load options; clang 23, one 4 x 64K run each against three reference runs) found none that improves decode or prefill. Four produce byte-identical GPU code, and one (the `iterative-ilp` scheduler on every source) crashes clang 23. Several cost a lot: no load-store vectorizer -19 % decode, unroll threshold 3000 -75 % prefill, the `iterative-maxocc` scheduler -53 % prefill, `-O2` -3.5 % prefill. The tree's flags are unchanged.

## Caveats

- Built and tested only on Cascade Lake + gfx906. ggml's CPU backend is built with `GGML_NATIVE`, so the CPU code **requires AVX-512 (with VNNI)** and will not run on hosts without it.
- ROCm 10.1 is a stream of the gfx906 apt repository built from AMD's TheRock development line; treat it as experimental.
- ROCm 10.1 has no `rocm-smi`; `amd-smi` replaces it (`amd-smi set -l HIGH`, `amd-smi set -o WATTS`).
- `GGML_HIP_RCCL=ON`: tensor split (`-sm tensor`) uses RCCL for prefill all-reduces. On the Mac Pro, RCCL needs the corrected XGMI topology (the firmware swaps the bridge links): about 25 GB/s all-reduce bus bandwidth without it, 34 with it; the `gfx906-rccl-topology` package generates it at boot.
- llama.cpp v1.0.0 removes API and options that were deprecated before (`--defrag-thold`, `llama_set_warmup` and others); scripts that still pass them stop with an unknown-argument error.
- With all four dies and a model whose layers share KV rows across devices (gemma4, mistral-large4), a sequence restored from a saved state can go wrong (the fork's KV replicas, in every release so far; `LLAMA_KV_REPLICA=0` avoids it). Qwen models are not affected.
- The ROCm 10.0 builds in the channel are unchanged (they still carry the global `-march=native`).

## Licensing

llama.cpp, ggml and mx-llama.cpp are MIT-licensed; see `/usr/share/doc/mx-llama-cpp-gfx906/copyright`.
