# mx-llama.cpp for gfx906 on ROCm 10.1, build r2: faster MTP decode at depth (Debian package)

`mx-llama-cpp-gfx906` is [exabit-io/mx-llama.cpp](https://github.com/exabit-io/mx-llama.cpp) `master` at
[`224b73e24`](https://github.com/exabit-io/mx-llama.cpp/commit/224b73e246811862a3d9f5d0af133bbbbd796515) (tag `gfx906/v1.0.0/r2/master`), built by Exabit, Inc. to run on
**ROCm 10.1.0-gfx906+20261018172427** (HIP 7.16.26385) for **gfx906** (Radeon VII, Instinct MI50/MI60, Radeon Pro Vega II).

Version `1.0.0+git20261116.224b73e24.rocm10.1.gfx906.20261018172427-1` replaces
`1.0.0+git20261115.290fc642a.rocm10.1.gfx906.20261018172427-1` (`-r1`) as the newest version in the rolling channel. It is
the same llama.cpp (v1.0.0), the same patches, the same compiler (ROCm 10.1's AMD clang 24) and the same build
configuration. **One number changed**: the compiler option that `-r1` introduced for the head-256 flash-attention tile
kernels goes from `-amdgpu-vgpr-threshold-percent=90` to `85`.

**Who gains.** Servers that run the MTP draft head (`--spec-type draft-mtp`) with two or three drafted tokens on a
head-256 model at long context. On Qwen3.8-27B Q8_0 at 118K tokens of context with `--spec-draft-n-max 3`, decode is
**6.9 % faster than `-r1`** (acceptance below). Prefill is 0.3 to 0.4 % faster. Plain decode is unchanged:
its kernel is the same code, instruction for instruction.

**What was wrong with 90.** A compiler option on one source file changes every kernel in that file, and the tile file
holds one kernel per batch shape: a stream that evaluates one token per step runs one kernel, two tokens another, three
or four a third, and so on up to the prefill kernel. `-r1`'s value was chosen on the prefill kernel and accepted with a
benchmark that runs only the prefill and the one-token kernels. In the three-or-four-token kernel, which is what
verifies three drafted tokens, 90 made the compiler reload 20 spilled registers from the stack in every iteration of
the kernel's one loop (without the option: 1). `-r1` was therefore 1.8 % slower at MTP decode at depth than the same
compiler with no option at all. At 85 no kernel of the file has a reload inside a loop.

It contains:

- [mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp), Marko Tombak's gfx906 fork of llama.cpp, up to `cdc81dae4` (the column-thread gated-delta-net kernel): q8 repack, custom all-reduce and tensor-parallel work, MMQ tile tables, DSpark/DFlash, Qwen3.8-Flash-Next and more.
- llama.cpp **v1.0.0** (b12359), merged into that fork, with two HIP fixes: the grouped top-k MoE fusion on 64-wide wavefronts, and `PAD_REFLECT_1D` launches of 2^32 or more threads. Offered to mxxm-t as [mxxm-t/mx-llama.cpp#24](https://github.com/mxxm-t/mx-llama.cpp/pull/24).
- **Exabit's gfx906 patches**: fused RMS norm + residual add + Q8_1 activations on the decode path, gated-delta-net producers folded into the GDN kernel, a 16-column MMVQ window, DPP warp reductions on GCN, an all-reduce size gate, the max-ilp scheduler, and the LLVM 24 rule for the head-256 tile kernels (changed in this build).

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

A machine that already has the channel gets this build with `sudo apt update && sudo apt upgrade`.

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
| Defaults of the tree | `GGML_HIP_RCCL=ON`, `GGML_CUDA_FA_QUANTS=all` (every flash-attention K/V quant pair), `GGML_TP_AR_MAX_NE=20481`, `GGML_HIP_GFX906_MAX_ILP=ON` (with clang 24 or newer this also sets `-mllvm -amdgpu-vgpr-threshold-percent=85` on the head-256 tile instance) |
| CPU target | CMake Release defaults; ggml's CPU backend with `GGML_NATIVE` for **Cascade Lake**: AVX512F/DQ/BW/VL/CD + AVX512-VNNI |
| Install prefix | `/opt/mx-llama.cpp` (`bin/`, `lib/`); every ELF carries `DT_RPATH=/opt/mx-llama.cpp/lib:/opt/rocm/core-10.1/lib` |
| Build host | Xeon W-3275M (Cascade Lake-W), Mac Pro 7,1, 4x Vega 20 |

Nothing is put on the default `PATH`, and no other llama.cpp/ggml installation can shadow the package's libraries:
`DT_RPATH` is searched before `LD_LIBRARY_PATH` and `ld.so.conf.d`. The same holds for ROCm: the binaries load ROCm 10.1
from `/opt/rocm/core-10.1/lib` whatever `/opt/rocm` points at. Run the tools from `/opt/mx-llama.cpp/bin`, for example
`/opt/mx-llama.cpp/bin/llama-server`.

## Verified on the build host

The installed package (`/opt/mx-llama.cpp`) was tested on 4x gfx906 (2x Radeon Pro Vega II Duo), with GPUs capped at 125 W and the generated RCCL topology in effect. The reference is the `-r1` package, tested the same way:

- **GPU code against `-r1`** (a source build of `-r1`'s commit by the same compiler, 10,735 device functions compared): 23 functions have other instructions, all in the head-256 tile object (22 tile kernels and the function that combines their results). 1,592 others differ in one address constant each and in no instruction (the name of the source directory is in the string pool, so strings move); the remaining 9,120 are byte-identical.
- **`test-backend-ops`** (the package does not ship the tests; the `-r1` source build's test binary with this package's GPU library preloaded, checked in each process's memory map): **20388/20388** on each of the four dies. Its 5,721 flash-attention cases include 165 with head size 256, which reach the batch shapes and head ratios that a Qwen3.8-27B run does not.
- **Perplexity** (Qwen3.8-27B Q8_0, wikitext-2, 16K x 6 chunks, `-sm tensor`): **5.6093 +/- 0.0622**, identical to `-r1`.
- **Output is byte-identical to `-r1`**: the greedy `-sm layer -lm dio` text and every MTP request (same SHA-256 of the text at `-np 1` and `-np 4`), with the same 154 of 201 drafts accepted. At 118K tokens of context the three 768-token answers of the acceptance cell are identical to `-r1`'s as well, with the same draft counts.
- **MTP** (`llama-server --spec-type draft-mtp --spec-draft-n-max 2`, `-sm tensor`), 256 tokens in steady state at 8K context: 73.2 / 73.1 / 73.3 tok/s at `-np 1` and 75.8 / 75.7 / 76.1 with `-np 4` and one active stream. Run back to back with a source build of `-r1`: 73.1 / 73.0 / 73.6 and 73.4 / 73.1 / 73.4 for this package, 73.1 / 72.9 / 73.4 and 73.3 / 73.1 / 73.5 for `-r1`; the first prompt (2,089 tokens) prefills at 731.3 and 731.2 tok/s against 730.7 and 730.3. At this depth the two builds are the same; the difference is at long context (next sections).
- **`-sm layer -lm dio`**: the model file is opened with `O_DIRECT`, no read falls back to buffered I/O.
- **Qwen3.8-Flash-Next** (qwen4exp, `-sm tensor --n-cpu-moe 41`): loads, and generation is coherent.
- **RCCL** 2.30.7 initialises and reads the generated topology (`/etc/rccl.conf`).
- **Build properties**: all 55 ELF files carry `DT_RPATH=/opt/mx-llama.cpp/lib:/opt/rocm/core-10.1/lib`. With `LD_LIBRARY_PATH` naming another llama.cpp build and ROCm 10.0's library directory, every ggml/llama library still comes from `/opt/mx-llama.cpp/lib` and all 26 ROCm libraries from `/opt/rocm/core-10.1`. `libggml-cpu` contains AVX-512 code (10,488 zmm and 1,189 `vpdpbusd` instructions).
- Not repeated for this build: the decode-path fold check, `test-llama-archs`, `test-fp8` and `test-llama-context` (results in the `-r1` release notes). The sources they exercise are unchanged, the host compiler and flags are the same, and the GPU code outside the one tile object is the same.

## The tile rule: 90 to 85

All measurements: Qwen3.8-27B Q8_0 on four dies (`-sm tensor`, f16 KV cache, 125 W per die, RCCL with the gated custom all-reduce), ROCm 10.1 runtime. In every table the arms differ only in the GPU library, and within it only in the one tile object.

**The kernels.** With this model (head size 256, two heads per block) a stream that evaluates one token in a step runs the tile kernel `<1,2>`, two tokens `<2,2>`, three or four `<4,2>`, five to eight `<8,2>`, nine or more `<16,2>`. Decode is `<1,2>`, prefill is `<16,2>`, and the verification of three drafted tokens is `<4,2>`. Spilled-register reloads that lie inside a loop, per kernel, and the loop's size in instructions:

| Tile object | `<1,2>` decode | `<2,2>` | `<4,2>` | `<8,2>` | `<16,2>` prefill |
|---|---|---|---|---|---|
| clang 23 (ROCm 10.0) | 0 / 856 | 0 / 735 | 1 / 1120 | 2 / 1035 | 0 / 1883 |
| clang 24, no option | 0 / 864 | 0 / 744 | 1 / 1138 | 2 / 1039 | 10 / 1906 |
| clang 24, threshold 90 (`-r1`) | 0 / 864 | 0 / 744 | **20 / 1168** | 0 / 1036 | 0 / 1894 |
| clang 24, threshold 85 (this build) | 0 / 864 | 0 / 744 | 0 / 1135 | 0 / 1038 | 0 / 1898 |

At 85 none of the file's 34 tile kernels has a reload in a loop. The option does not exist in clang 23, so the ROCm 10.0 builds in the channel cannot use it and are unchanged.

**MTP decode at depth.** `llama-server -np 1 -c 131072 --spec-type draft-mtp --spec-draft-n-max 3`; one chat request with a 117,740-token prompt (wikitext-2 plus a question), then two more questions on the same text; 768 generated tokens each, greedy. Decode is the mean rate of the three requests, prefill the rate of the long prompt. Three server starts per arm in shuffled order; all 15 produced the same text with the same draft counts (2,306 tokens drafted, 1,530 accepted), so the arms did the same work.

| Tile object | MTP decode at 117.7K, tok/s | Prefill, tok/s |
|---|---|---|
| clang 23 (ROCm 10.0) | 46.28 (46.25-46.30) | 608.5 (608.0-608.8) |
| clang 24, no option | 46.33 (46.27-46.46) | 594.2 (593.7-594.7) |
| clang 24, threshold 90 (`-r1`) | 45.48 (45.42-45.52) | 610.1 (610.0-610.3) |
| clang 24, threshold 85 (this build) | 48.36 (48.32-48.41) | 612.2 (612.0-612.3) |

Threshold 85 is 6.3 % above `-r1` and 4.5 % above ROCm 10.0's compiler, every run above every run. The second figure is a gain that no earlier build had: clang 23's kernel carries one reload in that loop too.

**The two design-point kernels** (`llama-batched-bench`, three runs per arm in shuffled order, mean and range), to make sure 85 gives back nothing that 90 had:

| Cell | Metric | clang 23 (ROCm 10.0) | threshold 90 (`-r1`) | threshold 85 (this build) |
|---|---|---|---|---|
| 4 x 64K | decode, tok/s per slot | 24.762 (24.755-24.776) | 24.744 (24.704-24.769) | 24.760 (24.749-24.782) |
| 4 x 64K | prefill, tok/s | 808.2 (807.7-809.0) | 809.3 (809.3-809.4) | 812.1 (811.4-812.4) |
| 1 x 128K | decode, tok/s | 39.965 (39.894-40.006) | 40.210 (40.157-40.245) | 40.123 (40.081-40.170) |
| 1 x 128K | prefill, tok/s | 604.3 (604.2-604.5) | 605.8 (605.7-605.9) | 608.2 (607.9-608.6) |

Prefill at 85 is above 90 in every run of both cells (+0.34 %, +0.40 %). Decode at 90 and 85 is the same code; the two differ by +0.07 % and -0.21 % with overlapping ranges.

## Acceptance of this package

**At the service's two design points**, 1,024 generated tokens, against a source build of `df2c6e5c5` compiled by and run on ROCm 10.0. Five runs per arm per cell, the arms alternating in a seeded random order. The rule was written before the first run: a difference counts if it holds at q < 0.10 (exact permutation test, Benjamini-Hochberg over the four tests) and is 2 % or more.

| Cell | Metric | ROCm 10.0 source build | This package | Effect | p / q |
|---|---|---|---|---|---|
| 4 x 64K | decode, tok/s per slot | 24.829 (24.80-24.89) | 24.728 (24.71-24.76) | -0.40 % | 0.008 / 0.016 |
| 4 x 64K | prefill, tok/s | 808.3 (807.7-808.8) | 811.0 (810.4-811.3) | +0.34 % | 0.008 / 0.016 |
| 1 x 255K | decode, tok/s | 32.035 (32.01-32.08) | 32.054 (31.99-32.06) | +0.06 % | 0.57 / 0.57 |
| 1 x 255K | prefill, tok/s | 411.1 (411.1-411.2) | 414.6 (414.5-414.8) | +0.84 % | 0.048 / 0.063 |

20 runs, none discarded (mean and range at 4 x 64K, median and range at 1 x 255K). No difference reaches 2 %: both cells are neutral under the rule and **the package is accepted**. Prefill is above the reference in every run of both cells. Decode at 4 x 64K is below it in every run, by 0.40 % (0.16 ms per step); that is the ROCm 10.1 runtime's cost (see "Unchanged from `-r1`"), which `-r1` showed as well (-0.52 % in its own acceptance). Decode at 1 x 255K does not separate.

**MTP at depth**, the cell above on the installed package: its own GPU library against `-r1`'s (and ROCm 10.0's compiler's tile object for reference), three server starts per arm in shuffled order. The rule was written before the first run: the package passes if its MTP decode is above `-r1`'s in every run by 3 % or more and its prefill is not below `-r1`'s in every run by more than 0.25 %.

| GPU library under the installed package's `llama-server` | MTP decode at 117.7K, tok/s | Prefill, tok/s |
|---|---|---|
| this package's own | 48.25 (48.22-48.30) | 610.2 (608.5-611.4) |
| `-r1`'s (threshold 90) | 45.14 (45.07-45.18) | 608.2 (607.9-608.6) |
| this build with the tile object from ROCm 10.0's clang 23 | 46.07 (46.04-46.12) | 607.2 (606.9-607.6) |

MTP decode is **6.9 % above `-r1`**, every run above every run, and 4.7 % above the clang 23 kernel: passed. Prefill is 0.32 % above `-r1` (two of three runs above every `-r1` run, one inside its range). All nine server starts generated the same three answers (SHA-256 of each) with the same draft counts, so the package's output at depth is `-r1`'s.

## Unchanged from `-r1`

- **Why the option exists at all.** ROCm 10.1's LLVM 24 compiles the head-256 prefill kernel with ten register reloads in its inner loop (prefill -4.1 % at one 255K-token stream). The cause is a compiler defect that LLVM has since fixed (machine LICM's register-pressure limit wraps around on AMDGPU; the fix is not in ROCm 10.1's compiler). The scheduler option removes the reloads, and on a compiler with the fix it is still worth keeping. The `-r1` release notes have the analysis.
- **The ROCm 10.1 runtime's decode cost** (0.3 to 0.5 % at 4 x 64K against ROCm 10.0's libraries, same code) is in ROCr, the HSA runtime: ROCm 10.0's ROCr polls its asynchronous events in a loop that never sleeps and keeps one host core busy; ROCm 10.1's naps between polls. The package changes nothing there.
- **No global `-march=native`** on the host code; ggml's CPU backend is built for the native CPU (`GGML_NATIVE`).

## Caveats

- Built and tested only on Cascade Lake + gfx906. ggml's CPU backend is built with `GGML_NATIVE`, so the CPU code **requires AVX-512 (with VNNI)** and will not run on hosts without it.
- The MTP figures are for one model (head size 256, two heads per attention block), one depth and three drafted tokens. With one drafted token (`--spec-draft-n-max 1`) verification runs the two-token kernel, which is the same code in `-r1` and this build.
- ROCm 10.1 is a stream of the gfx906 apt repository built from AMD's TheRock development line; treat it as experimental.
- ROCm 10.1 has no `rocm-smi`; `amd-smi` replaces it (`amd-smi set -l HIGH`, `amd-smi set -o WATTS`).
- `GGML_HIP_RCCL=ON`: tensor split (`-sm tensor`) uses RCCL for prefill all-reduces. On the Mac Pro, RCCL needs the corrected XGMI topology (the firmware swaps the bridge links): about 25 GB/s all-reduce bus bandwidth without it, 34 with it; the `gfx906-rccl-topology` package generates it at boot.
- llama.cpp v1.0.0 removes API and options that were deprecated before (`--defrag-thold`, `llama_set_warmup` and others); scripts that still pass them stop with an unknown-argument error.
- With all four dies and a model whose layers share KV rows across devices (gemma4, mistral-large4), a sequence restored from a saved state can go wrong (the fork's KV replicas, in every release so far; `LLAMA_KV_REPLICA=0` avoids it). Qwen models are not affected.

## Licensing

llama.cpp, ggml and mx-llama.cpp are MIT-licensed; see `/usr/share/doc/mx-llama-cpp-gfx906/copyright`.
