# mx-llama.cpp for gfx906 on ROCm 10.0, rebuilt without the global `-march=native` (Debian package)

`mx-llama-cpp-gfx906` is [exabit-io/mx-llama.cpp](https://github.com/exabit-io/mx-llama.cpp) `master` at
[`290fc642a`](https://github.com/exabit-io/mx-llama.cpp/commit/290fc642ac23fc8ec73c745f40cd315c7ff32d8c) (tag `gfx906/v1.0.0/r1/master`), built by Exabit, Inc. against
**ROCm 10.0.0-gfx906+20261018172427** (HIP 7.15.26333) for **gfx906** (Radeon VII, Instinct MI50/MI60, Radeon Pro Vega II).

Version `1.0.0+git20261115.290fc642a.rocm10.0.gfx906.20261018172427-1` is for machines that stay on ROCm 10.0. It
replaces `1.0.0+git20261114.df2c6e5c5.gfx906.20261018172427-1` for them. **It is not the newest version in the rolling
channel**: that is the ROCm 10.1 build of the same commit, which `apt upgrade` installs. Install this one by naming it:

```sh
sudo apt install --allow-downgrades mx-llama-cpp-gfx906=1.0.0+git20261115.290fc642a.rocm10.0.gfx906.20261018172427-1
sudo apt-mark hold mx-llama-cpp-gfx906
```

It is the same llama.cpp (v1.0.0) and the same patches as the earlier ROCm 10.0 build. What changed:

1. **The host code is no longer compiled with a global `-march=native`.** On `libllama` and `libllama-common` that flag
   cost 0.4 % of decode at 4 slots x 64K (the auto-vectorized loops it enables are slower there). ggml's CPU backend is
   still built for the native CPU (`GGML_NATIVE`), so the AVX-512 requirement is unchanged. Every earlier build in the
   channel that runs on ROCm 10.0 (0.7.0, 0.9.0, the first 1.0.0) carries the flag.
2. **The binaries name their ROCm.** Every ELF now carries `DT_RPATH=/opt/mx-llama.cpp/lib:/opt/rocm/core-10.0/lib`.
   The earlier ROCm 10.0 builds found ROCm through `/opt/rocm`, so on a machine that also has ROCm 10.1 installed they
   load 10.1's libraries. This build loads ROCm 10.0 whatever `/opt/rocm` points at.
3. The source commit is one ahead (`290fc642a`): a compiler option for LLVM 24 on one flash-attention source file. ROCm
   10.0's compiler is LLVM 23 and does not take that path: every kernel is byte-identical to a ROCm 10.0 build of the
   earlier commit.

It contains:

- [mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp), Marko Tombak's gfx906 fork of llama.cpp, up to `cdc81dae4` (the column-thread gated-delta-net kernel): q8 repack, custom all-reduce and tensor-parallel work, MMQ tile tables, DSpark/DFlash, Qwen3.8-Flash-Next and more.
- llama.cpp **v1.0.0** (b12359), merged into that fork, with two HIP fixes: the grouped top-k MoE fusion on 64-wide wavefronts, and `PAD_REFLECT_1D` launches of 2^32 or more threads. Offered to mxxm-t as [mxxm-t/mx-llama.cpp#24](https://github.com/mxxm-t/mx-llama.cpp/pull/24).
- **Exabit's gfx906 patches**: fused RMS norm + residual add + Q8_1 activations on the decode path, gated-delta-net producers folded into the GDN kernel, a 16-column MMVQ window, DPP warp reductions on GCN, an all-reduce size gate, the max-ilp scheduler, and the LLVM 24 rule for the head-256 tile kernels (inactive in this build).

## Install with apt

The rolling channel holds every version; one source line:

```sh
sudo install -d -m 755 /etc/apt/keyrings
curl -fsSL https://exabit-io.github.io/resize-amdgpu-bars/keys/resize-amdgpu-bars.gpg \
    | sudo tee /etc/apt/keyrings/resize-amdgpu-bars.gpg > /dev/null
echo 'deb [signed-by=/etc/apt/keyrings/resize-amdgpu-bars.gpg] https://github.com/exabit-io/resize-amdgpu-bars/releases/download/mx-llama-cpp-gfx906 ./' \
    | sudo tee /etc/apt/sources.list.d/mx-llama-cpp-gfx906.list
sudo apt update
sudo apt install --allow-downgrades mx-llama-cpp-gfx906=1.0.0+git20261115.290fc642a.rocm10.0.gfx906.20261018172427-1
```

This release is also a signed flat apt repository of its own (same key), for pinning exactly this version. The `.deb` carries a `debsigs` origin signature.

It needs a working **ROCm 10.0 gfx906** stack from the [gfx906 apt repository](https://github.com/mixa3607/ML-gfx906).
The package depends only on the runtime pieces it links: `amdrocm-runtime10.0`, plus the rocBLAS and RCCL host
libraries and their gfx906 kernel packages.

## Build configuration

| | |
|---|---|
| ROCm at run time | 10.0.0-gfx906+20261018172427 (HIP 7.15.26333, RCCL 2.30.4) |
| GPU compiler | ROCm 10.0.0-gfx906+20261018172427's `clang++` (AMD clang 23.0.0git); recorded in `/usr/share/doc/mx-llama-cpp-gfx906/BUILD-INFO` |
| GPU arch | `AMDGPU_TARGETS=gfx906` |
| Defaults of the tree | `GGML_HIP_RCCL=ON`, `GGML_CUDA_FA_QUANTS=all` (every flash-attention K/V quant pair), `GGML_TP_AR_MAX_NE=20481`, `GGML_HIP_GFX906_MAX_ILP=ON` |
| CPU target | CMake Release defaults; ggml's CPU backend with `GGML_NATIVE` for **Cascade Lake**: AVX512F/DQ/BW/VL/CD + AVX512-VNNI |
| Install prefix | `/opt/mx-llama.cpp` (`bin/`, `lib/`); every ELF carries `DT_RPATH=/opt/mx-llama.cpp/lib:/opt/rocm/core-10.0/lib` |
| Build host | Xeon W-3275M (Cascade Lake-W), Mac Pro 7,1, 4x Vega 20 |

Nothing is put on the default `PATH`, and no other llama.cpp/ggml installation can shadow the package's libraries:
`DT_RPATH` is searched before `LD_LIBRARY_PATH` and `ld.so.conf.d`. The same holds for ROCm: the binaries load ROCm 10.0
from `/opt/rocm/core-10.0/lib` whatever `/opt/rocm` points at. Run the tools from `/opt/mx-llama.cpp/bin`, for example
`/opt/mx-llama.cpp/bin/llama-server`.

## Verified on the build host

The installed package (`/opt/mx-llama.cpp`) was tested on 4x gfx906 (2x Radeon Pro Vega II Duo), with GPUs capped at 125 W and the generated RCCL topology in effect. The machine has ROCm 10.0 and 10.1 installed side by side, with `/opt/rocm` pointing at 10.1:

- **Perplexity** (Qwen3.8-27B Q8_0, wikitext-2, 16K x 6 chunks, `-sm tensor`): **5.6093 +/- 0.0622**, identical to every other 1.0.0 build.
- **Output is byte-identical to the ROCm 10.1 package of the same commit**: the greedy `-sm layer -lm dio` text, the Qwen3.8-Flash-Next text, and every MTP request (same SHA-256 of the text at `-np 1` and `-np 4`), with the same 154 of 201 drafts accepted.
- **`test-backend-ops`** (the package does not ship the tests; a source build of the previous commit by the same compiler, whose kernels are byte-identical to the package's, tested on 2026-11-14): **20388/20388** on each of two dies.
- **MTP** (`llama-server --spec-type draft-mtp --spec-draft-n-max 2`, `-sm tensor`), 256 tokens in steady state: **74.2 / 74.1 / 74.7 tok/s** at `-np 1` and 76.8 / 76.8 / 76.6 with `-np 4` and one active stream. The first prompt (2,089 tokens) prefills at 736 tok/s (one sample). These are 8K smoke values; the comparison at the service's design points is the next section.
- **`-sm layer -lm dio`**: the model file is opened with `O_DIRECT`, no read falls back to buffered I/O.
- **Qwen3.8-Flash-Next** (qwen4exp, `-sm tensor --n-cpu-moe 41`): loads, and generation is coherent.
- **RCCL** (2.30.4) initialises in every tensor-split run: no all-reduce probe failure in any log.
- **Build properties**: all 55 ELF files carry `DT_RPATH=/opt/mx-llama.cpp/lib:/opt/rocm/core-10.0/lib`. With `LD_LIBRARY_PATH` naming another llama.cpp build and ROCm 10.1's library directory, every ggml/llama library still comes from `/opt/mx-llama.cpp/lib` and all 26 ROCm libraries from `/opt/rocm/core-10.0`. `libggml-cpu` contains AVX-512 code (10,488 zmm and 1,189 `vpdpbusd` instructions).
- **GPU code**: of the 10,323 device functions in the 215 HIP objects, 10,157 are byte-identical to those of a ROCm 10.0 build of the previous commit `df2c6e5c5`, every kernel among them. The 166 that differ are copies of the error printer `no_device_code`, which embeds the path of the source file as it was on the build machine, and of the print helper inlined beside it.

## Acceptance at the design points

Qwen3.8-27B Q8_0 on four dies (`-sm tensor`, f16 KV cache, 125 W per die, RCCL with the gated custom all-reduce), `llama-batched-bench`, 1,024 generated tokens, against a source build of `df2c6e5c5` compiled by and run on ROCm 10.0 (no global host flag). Five runs per arm per cell, the arms alternating in a seeded random order. The rule was written before the first run: a difference counts if it holds at q < 0.10 (exact permutation test, Benjamini-Hochberg over the four tests) and is 2 % or more.

| Cell | Metric | ROCm 10.0 source build | This package | Difference | p / q |
|---|---|---|---|---|---|
| 4 x 64K | decode, tok/s per slot | 24.857 (24.81-24.89) | 24.855 (24.82-24.88) | -0.01 % | 0.87 / 1.0 |
| 4 x 64K | prefill, tok/s | 808.1 (807.7-808.3) | 807.7 (806.3-808.5) | -0.04 % | 0.56 / 1.0 |
| 1 x 255K | decode, tok/s | 32.057 (32.01-32.14) | 32.042 (31.99-32.11) | -0.05 % | 1.0 / 1.0 |
| 1 x 255K | prefill, tok/s | 411.0 (410.9-411.2) | 410.9 (410.8-411.2) | -0.03 % | 0.38 / 1.0 |

Values are the mean of the five runs at 4 x 64K and the median at 1 x 255K, with the range of the runs; 20 runs, none discarded. No difference reaches 0.1 % and the ranges overlap in every row: the package is accepted. Both arms run on ROCm 10.0, the package's kernels are the reference build's, and the reference build has no global host flag either, so this is the result to expect. The earlier ROCm 10.0 build's global `-march=native` was measured at 0.4 % of decode at 4 x 64K (three runs per arm, host libraries exchanged under one GPU library).

## Caveats

- Built and tested only on Cascade Lake + gfx906. ggml's CPU backend is built with `GGML_NATIVE`, so the CPU code **requires AVX-512 (with VNNI)** and will not run on hosts without it.
- ROCm 10.0 is an unannounced stream in the gfx906 apt repository with no upstream counterpart; treat it as experimental.
- `apt upgrade` moves an unheld installation to the channel's newest version, the ROCm 10.1 build (ROCm streams install side by side; apt adds the 10.1 runtime packages and removes nothing). Hold the package to stay on ROCm 10.0.
- `GGML_HIP_RCCL=ON`: tensor split (`-sm tensor`) uses RCCL for prefill all-reduces. On the Mac Pro, RCCL needs the corrected XGMI topology (the firmware swaps the bridge links): about 25 GB/s all-reduce bus bandwidth without it, 34 with it; the `gfx906-rccl-topology` package generates it at boot.
- llama.cpp v1.0.0 removes API and options that were deprecated before (`--defrag-thold`, `llama_set_warmup` and others); scripts that still pass them stop with an unknown-argument error.
- With all four dies and a model whose layers share KV rows across devices (gemma4, mistral-large4), a sequence restored from a saved state can go wrong (the fork's KV replicas, in every release so far; `LLAMA_KV_REPLICA=0` avoids it). Qwen models are not affected.
- The 0.7.0 and 0.9.0 builds in the channel are unchanged (global `-march=native`, ROCm through `/opt/rocm`).

## Licensing

llama.cpp, ggml and mx-llama.cpp are MIT-licensed; see `/usr/share/doc/mx-llama-cpp-gfx906/copyright`.
