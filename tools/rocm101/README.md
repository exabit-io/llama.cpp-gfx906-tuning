# ROCm 10.0 -> 10.1 on macpro2019-01, 2026-11-14

Record of the migration (lead: "upgrade to the latest and rebuild"). Report for readers:
`reports/2026-11-14-rocm-10.1.md` in the tuning repository (`/root/llama.cpp-benchmarking`), which also holds a copy of this
file and of the scripts in `tools/rocm101/` and the result lines in `data/raw/2026-11-14/rocm101/`. The logs named below
are in `/root/rocm101-migration-20261114/` on the box. Previous migration: `/root/rocm10-migration-20260919/`.

## What was done to the system

| Step | How | Log |
|---|---|---|
| state before | packages, alternatives, manual marks, sonames, hipconfig, rocminfo | `packages-before.tsv`, `alternatives-before.txt`, `manual-before.txt`, `sonames-10.0.txt`, `hipconfig-10.0.txt`, `rocminfo-10.0.txt`, `preferences.d-before/` |
| apt pin | `/etc/apt/preferences.d/rocm-gfx906`: `amdrocm*` pinned to `10.1.0-gfx906*` (was `10.0.0-gfx906*`) | the file's own comment |
| install 10.1 | the 51 manually installed 10.0 package names with 10.1 in the name (`want-10.1.txt`): 64 new packages, nothing removed | `apt-simulate.txt`, `install-10.1.log` |
| unversioned names | `amdrocm-amdsmi`, `-blas`, `-llvm`, `-rand`, `-runtime` upgraded to 10.1.0 | `install-10.1-unversioned.log` |
| alternatives | switched by the packages themselves (auto mode, 10.1 has the higher priority): 26 links on 10.1; `rocm-smi` stays on 10.0 because 10.1 has none | `alternatives-after.txt` |
| test environment | `/root/rocm-tests/bench/gpu-test-env.sh`: `rocm-smi --setperflevel` replaced by a sysfs write (backup `.pre-rocm101`) | |
| build chroot | `mmdebstrap` installed; noble chroot `/var/cache/exabit-backports/noble-buildd` (for the sympy backport) | `install-mmdebstrap.log`, `/root/backports-noble/` |

ROCm 10.0 is still installed (`/opt/rocm/core-10.0`). Until 2026-11-15 its compiler built the llama package; since the
`-r1` package (master 290fc642a) the package is compiled by 10.1, and 10.0 is kept for the reference builds
(`/root/build-master-v100r` is the reference arm of every acceptance) and for sources without the LLVM 24 tile rule.
Rollback of the stream: set the pin back to `10.0.0-gfx906*`, `apt-get install` the five unversioned names at 10.0.0, and
`update-alternatives --set` the links (or remove the 10.1 packages; the alternatives fall back by themselves).

## What ROCm 10.1 is

`10.1.0-gfx906+20261018172427` (the same build stamp as the 10.0 stream's current build): HIP 7.16.26385 (10.0: 7.15.26333),
AMD clang 24.0.0git (10.0: 23.0.0git), RCCL 2.30.7 (10.0: 2.30.4), amd_smi 27.1.0. Library sonames are those of 10.0, minus
`liboam.so.1` and `librocm_smi64.so.1` (no `rocm-smi`; `amd-smi` replaces it), plus four new ones (`sonames-10.1.txt`).

## Measurements

| Script | What | Result |
|---|---|---|
| `suite-101.sh` | correctness of master df2c6e5c5 compiled by and run on 10.1 | all as on 10.0 |
| `rocm-ab.sh`, `rocm-ab-stats.py` | 10.0 against 10.1 at 4x64K and 1x255K, n=5, rule fixed before data | decode unchanged; prefill -1.39 % and -4.13 %: 10.1-compiled build REJECTED |
| `rocm-cross.sh` | compiler or runtime? the two crossed arms | the prefill loss is the compiler's; the 10.1 runtime leaves prefill alone (its decode cost: `pkg-swap.sh` below) |
| `suite-mixed.sh` | correctness of 10.0-compiled code on the 10.1 runtime | all as on 10.0 |
| `codegen/` (`spilldiff.py`, `metacmp.py`, `trycc.py`, `tileexp.py`, `swaplink.py`), `screen.sh` | which kernel, and can a setting fix it | one object: the head-256 flash-attention tile kernel; no scheduler strategy or table setting tried restores it |
| `deb-test-llama-101.sh`, `rocm-ab2.sh` | the package as published (10.0 compiler, 10.1 runtime), n=5 per arm | ACCEPTED: prefill = 10.0 (805.7 / 410.2 tok/s), decode -0.97 % at 4x64K and -0.74 % at 1x255K, under the 2 % floor |
| `publish-llama-101.sh`, `dogfood-llama-101.sh` | release `mx-llama-cpp-1.0.0-gfx906-rocm10.1`, rolling channel, reinstall from the channel | published 2026-11-15 00:12 UTC; chain verified |
| `pkg-swap.sh` (three sessions) | where the package's decode difference is: its libraries swapped for a source build's by `LD_PRELOAD` | 0.6 % the ROCm 10.1 runtime, 0.4 % the deb's `-march=native` on `libllama`/`libllama-common` |
| `flag-builds.sh`, `flag-screen.sh` | other host compiler flags (lead's question): skylake-avx512, x86-64-v4, `-mtune=intel`, no vectorizer, `-O2`, no semantic interposition, LTO, PGO | n = 3 per arm, 33 cells, none discarded: no arm beats no flags (24.672); `-march=native` -0.41 %, the 512-bit arms -0.62 and -0.77 %, `-O2` -1.29 %; vectorizer off = no flags; LTO and PGO gain nothing (report, section 5.2) |
| `rt-swap.sh` | which ROCm 10.1 library carries the runtime's decode cost: every ROCm library preloaded by full path, one group (RCCL, HIP, ROCR, BLAS stack, comgr, the rest) taken from 10.0 | n = 3 per arm, 27 cells, none discarded: all-10.0 24.826, all-10.1 24.748 (-0.32 %); only ROCr (`libhsa-runtime64`) from 10.0 restores it (24.810). With `PERF=1`: 10.0's `Runtime::AsyncEventsLoop` polls without sleeping (35.9 % of the bench's cycles), 10.1's sleeps between scans (report, section 5.3) |
| `codegen/asmcmp.py`, `codegen/memflow.py` | one kernel in two code objects: instruction mix, loops, how loads overlap their waits | head-256 tile kernel: same instruction mix under LLVM 23 and 24; LLVM 24 reloads 10 spilled registers inside the hot loop (LLVM 23: none) and adds 4 memory waits there |
| `drv-test.sh`, `codegen/drvexp.py` | is it the HIP driver's new offload pipeline (device code through LTO)? one tile object swapped, 1x128K, n = 2 | no: clang 23 605.3-605.7 tok/s and clang 24 590.3-590.7 with it or without |
| `tile2-build.sh`, `tile2-screen.sh`, `codegen/ircross.py`, `codegen/loopstat.py` | middle end or code generator (each compiler's optimized bitcode through the other's code generator), then single code-generator options on the tile object | the code generator: clang 23 bitcode through LLVM 24 590.9, clang 24 bitcode through LLVM 23 606.0; `-amdgpu-vgpr-threshold-percent=90` on the object 606.4 with no spills in the prefill kernel; `-amdgpu-use-amdgpu-trackers=1` costs decode 5 % (report, section 4.1) |
| `gpu-opt-builds.sh`, `gpu-opt-builds-rest.sh`, `gpu-opt-screen.sh` | survey of 21 single GPU-side compile options with the 10.0 compiler: builds, code comparison against the reference, one 4x64K sample each against three reference samples and a positive control | none faster; 4 give byte-identical GPU code, 1 does not build, several far slower (report, section 5.4) |
| `gpu-opt-confirm.sh` | the package's compiler: reference, plain 10.1 build, 10.1 with the rule (only `libggml-hip.so` differs); n = 3 shuffled at 4x64K and 1x128K, rule in the header | the candidate qualifies: worst effect -0.05 %, prefill +0.17 % and +0.27 % (report, section 5.5) |
| `suite-t90.sh`, `deb-test-llama-101.sh`, `rocm-ab3.sh` | master 290fc642a compiled by 10.1: correctness suite, package test, design-point acceptance of the installed package (n = 5 per arm) | every value and hash as before; ACCEPTED: prefill +0.27 % / +0.28 %, decode -0.52 % at 4x64K and +0.05 % at 1x255K (report, section 5.6) |
| `publish-llama-r1.sh`, `dogfood-llama-101.sh` | release `mx-llama-cpp-1.0.0-gfx906-rocm10.1-r1`, rolling channel, reinstall from the channel | published 2026-11-16 00:15 UTC; chain verified |
| `publish-torch-101.sh`, `/root/pytorch-gfx906-rocm10.1-2.15.0/`, `/root/backports-noble/` | PyTorch 2.15.0 for ROCm 10.1 and the `python3-sympy` 1.14.0 backport | published 2026-11-14 into the channel `python3-torch-gfx906` |
| `pkg-resid.sh` | is any of the `-r1` package's 4x64K decode difference its own? reference on 10.0, the package's program with the reference's libraries on 10.1, the package; n = 5, rule fixed before data | NO: runtime 0.49 %, package -0.08 % (overlapping); stage 2 not run; closed (report 5.7) |
| `gpu-opt24-builds.sh`, `gpu-opt24-builds-all.sh`, `gpu-opt24-screen.sh` | the GPU compile-option survey again with the shipping compiler (10.1's clang 24) at BOTH design points; controls with a known sign in each pass | in progress 2026-11-16 (report 5.8 when done) |
| `llvm-bisect/` (`prep.sh`, `step.sh`, `run.sh`, `confirm.sh`, `fixcheck.sh`, `bisect-screen.sh`, `fixrule-screen.sh`), `codegen/llccross.py` | which LLVM change makes the head-256 tile kernel spill: `llc` built per commit in `/root/llvm-bisect`, clang 23's bitcode compiled, reloads in the hot loop counted; then the objects timed | first bad commit `80df471b75d0` (machine LICM's pressure limit wraps on AMDGPU); upstream fix `1d7d16b94c61` is not in ROCm 10.1's release branch; timing confirms (604.1 / 588.7 controls; parent 600.6, first bad 589.3, revert 600.5, fix 601.0); the tree's rule adds 1.0 % on a fixed compiler and stays (report 4.2, 4.3). `llvm-bisect/ISSUE-DRAFT.md` + `repro/`: report for ROCm, not filed |
| `deb-test-llama-100.sh`, `rocm-ab4.sh`, `publish-llama-100r1.sh`, `dogfood-llama-100.sh` | the ROCm 10.0-stream package rebuilt from 290fc642a without the global host flag: package test, acceptance against the reference on the same runtime, release, install by version from the channel and back to the newest | DONE 2026-11-16: every kernel byte-identical to the reference build; package test clean (PPL 5.6093, MTP 154/201, hashes equal to the 10.1 package, 26 of 26 ROCm libraries from core-10.0 whatever the environment names); acceptance n = 5 neutral on both axes (-0.01 % / -0.04 % at 4 x 64K, -0.05 % / -0.03 % at 1 x 255K); release `mx-llama-cpp-1.0.0-gfx906-rocm10.0-r1`, in the channel below the 10.1 build; dogfooded; box back on the 10.1 `-r1` package (report 5.9) |
| `codegen/loopscan.py`, `codegen/tilescan.py`, `tilethr/` | static: scratch reloads inside loops in every device function, clang 23 against clang 24 (10,323 functions); the tile object's 34 kernels at every VGPR threshold | 10,056 unchanged, fewer in 216, more in 51; the FP8-V vec kernels 146 -> 623 (not the LICM bug: stays with `-disable-machine-licm` and every threshold); **threshold 90 puts 20 reloads into the loop of `<4,2>`, `<2,4>`, `<1,8>`**; 88 and 85 leave none in any kernel (report 5.10; `loopscan-c23-c24.txt`, `tilethr/tilescan-thresholds.txt`) |
| `mtp-depth.sh` (no decode data: raw prompt ended at 1 token), `mtp-depth2.sh` | MTP decode at 117.7K depth, `--spec-draft-n-max 3` (verification runs `<4,2>`), five tile objects under the installed `llama-server`, n = 3 shuffled, rule in the header | clang 23 46.28, clang 24 46.33, threshold 90 (`-r1`) **45.48 (-1.84 %)**, 88 48.59, 85 48.36 tok/s; prefill 608.5 / 594.2 / 610.1 / 611.4 / 612.2; rule's outcome: replace 90 by 85 (report 5.11) |
| `tile-thr-confirm.sh` | the replacement at the two design-point kernels: reference, 90, 85, 88 (only `libggml-hip.so` differs), n = 3 shuffled at 4x64K and 1x128K, rule in the header | 85 PASSES and is taken: prefill +0.34 % / +0.40 % over 90 (every sample), +0.48 % / +0.65 % over clang 23; decode level against both at 4x64K, +0.40 % over clang 23 at 1x128K; fork commit `224b73e24` (branch `hip-llvm24-tile-vgpr85`, worktree `/root/wt-master-t85`) (report 5.11) |
| `hostcc-builds.sh`, `hostcc-screen.sh` | another HOST compiler (section 5.2's open question): gcc 13.3 with and without OpenMP, Ubuntu clang 19, ROCm 10.1's clang 24; six host libraries preloaded under the installed `-r1` package, 4x64K, n = 3 shuffled, rule in the header | NO: clang 19 24.587 (-0.70 %), clang 24 24.556 (-0.82 %) against gcc 24.760, every run below both gcc arms; OpenMP worth 0.06 %; prefill equal; gcc 13 stays (report 5.13). Needs the package of the SAME commit installed: run it before installing a new deb |
| `fp8-kv.sh` (`screen.sh` with `KVK`/`KVV`/`ARMKV`) | an FP8 KV cache (`f8_e4m3`/`f8_e4m3`) at 4x64K: clang 23 against the `-r1` GPU code, and FP8 against f16; n = 3 shuffled, rule in the header | clang 24 costs FP8 nothing (decode 13.135 against 13.143, level; prefill +0.15 %); **FP8 halves decode: 13.13 against 24.76 tok/s per slot (-46.9 %)**, prefill +0.76 %; no option sought, not a throughput configuration; quality not measured (report 5.12) |
| `/root/mx-llama-deb/build-mx-llama-deb.sh /root/wt-master-t85`, `codegen/maskcmp.py`, `deb-test-llama-101.sh deb-test-llama-r2`, `smoke-r2-prefill/` | package `-r2` (master 224b73e24: threshold 85): build, GPU code against the `-r1` source build, package test (installs the deb) | 10,735 device functions: 23 with other instructions, all in the head-256 tile object; 1,592 differ in one `s_add_u32` immediate (source directory name in the string pool); 9,120 identical. Package test clean: PPL 5.6093, every hash equal to `-r1`'s, MTP 154/201, 55 ELF with the RPATH, 0 of 26 ROCm libraries outside core-10.1. Its 8K first-prompt prefill read 722 against `-r1`'s 749 of the day before; back to back 731.3 / 731.2 against 730.7 / 730.3: session, not package |
| `mtp-accept.sh` | acceptance part 1, MTP at depth on the INSTALLED package: its own GPU library, `-r1`'s, the clang 23 tile object; n = 3 shuffled, rule in the header (decode above `-r1` in every run by 3 % or more) | PASSED: 48.250 (48.222-48.301) against `-r1` 45.140 (45.073-45.183) = +6.89 %, clang 23 46.067; prefill 610.2 / 608.2 / 607.2; same text and draft counts in all 9 cells (report 5.11) |
| `rocm-ab5.sh` (`rocm-ab-stats.py`) | acceptance part 2, the two design points on the installed package against `/root/build-master-v100r` on 10.0; n = 5 per arm per axis, the survey's rule | ACCEPTED, both axes neutral: 4 x 64K decode 24.728 against 24.829 (-0.40 %, q 0.016), prefill 811.0 against 808.3 (+0.34 %); 1 x 255K decode 32.054 against 32.035 (+0.06 %), prefill 414.6 against 411.1 (+0.84 %, q 0.063); 20 runs, none discarded (report 5.11) |
| `tbo-r2.sh` | `test-backend-ops` on the package's GPU code: the `-r1` source build's test binary with the installed `libggml-hip.so` preloaded, one process per die, library read back from `/proc/<pid>/maps` | PASSED: 20388/20388 on each of four dies; 5,721 `FLASH_ATTN_EXT` cases, 165 with head size 256 |
| `publish-llama-r2.sh`, `dogfood-llama-101.sh` | fork `master` = `gfx906-single` = `gfx906-multi` = 224b73e24 and tag `gfx906/v1.0.0/r2/master` pushed; release `mx-llama-cpp-1.0.0-gfx906-rocm10.1-r2`, rolling channel (7 versions), reinstall from the channel | published 2026-11-16 19:22 UTC; chain verified (InRelease, Packages hash, deb hash, origin signature, identical to the local build). `dogfood-llama-101.sh` wants an ABSOLUTE output directory: a relative one lost every log line after its `cd` and printed false MISMATCH lines (`dogfood-llama-r2.relpath-attempt/`); fixed in the script (backup `.pre-abs`) |

Builds: `/root/build-master-v100r` (10.0, the reference of every comparison), `/root/build-master-v101` (10.1),
`/root/build-master-v100rn` (reference + `-march=native` host flags), `/root/build-flag-*` (host-flag variants),
`/root/build-v101-t90` (master 290fc642a by 10.1: the GPU code of the `-r1` package), `/root/build-gpuopt-*` (the survey's
variants), `/root/build-v101-trk` (the trackers option on every source: a record, never to be shipped),
`/root/build-hostcc-{gccnoomp,c19,c24}` (host-compiler variants of 290fc642a), `/root/build-gpuopt24-*` (the clang 24
survey's variants), `/root/swap/{c23,c24,v90,v88,v85}` (GPU libraries that differ only in the head-256 tile object:
the ROCm 10.1 build's objects with that one exchanged by `codegen/drvexp.py`). Source worktrees: `/root/wt-master-v100`
(df2c6e5c5), `/root/wt-master-t90` (290fc642a, `-r1`), `/root/wt-master-t85` (224b73e24, `-r2`; its package build tree is
`/root/mx-llama-deb/build`, and `/root/swap/{pkg,r1}` are the two packages' GPU libraries for `mtp-accept.sh`). The
cross-over builds, the scheduler variants of the tile kernels and the object-swap libraries were scratch and are
rebuilt by their scripts (`rocm-cross.sh`, `screen.sh`, `codegen/swaplink.py`).

## How these measurements are run (for the next agent)

- **One GPU job at a time, and nothing heavy on the host beside it.** A timing cell uses all four dies; builds, `perf
  report`, `fio`, a large `git clone` or an apt install on the host move decode by tenths of a percent (the runtime's
  0.3 % is one host thread's sleep). Every script here refuses to start while `llama-batched-bench` runs
  (`pgrep -x llama-batched-b`). Do not test with `pgrep -f NAME` from a command line that contains NAME: it matches
  its own shell.
- **Cells and their cost** (model load included): 4 x 64K with `-ntg 1024` 6.3 min; 1 x 128K with `-ntg 256` 4 min;
  1 x 255K (`260864`) with `-ntg 1024` 11.3 min. 64K is the floor for any result; smaller depths are smoke tests.
- **Every session carries its own reference arm.** The same build read 24.67, 24.79 and 24.87 tok/s per slot in
  sessions days apart (boots, cards added, kernel parameters). Values from two sessions are never subtracted; effects
  are, with that caveat stated.
- **The reading rule is written into the script header before the first cell** (what counts as up, down, level; what
  each outcome leads to). A one-sample pass needs a positive control whose sign is known (`schdef` in the survey).
- **Screens**: `screen.sh TAG DEPTH SLOTS NTG N ARM=BUILD:ROCMTREE[:LIBGGML-HIP] ...` (`SHUFFLE=1` for a fresh random
  order per block). An arm is a build's programs on a ROCm tree, optionally with another `libggml-hip.so` preloaded;
  this is how compiler variants are compared with everything else held fixed. Output `screen-TAG.tsv`.
- **Library swaps** (`pkg-swap.sh`, `pkg-resid.sh`, `rt-swap.sh`): preload ALL of a program's libraries by full path
  with one group taken from another build or ROCm tree; the loaded set is read back from `/proc/<pid>/maps` in every
  cell. Resolves 0.3-0.4 % at n = 3-4.
- **Before timing a compiler variant, compare its code**: `codegen/devcmp.py REF BUILD` (which GPU objects differ),
  `codegen/fncmp.py` (which functions), `codegen/loopstat.py` (spill reloads inside loops). Byte-identical GPU code is
  not timed. `codegen/trycc.py` recompiles one source with extra flags; `codegen/swaplink.py` relinks the GPU library
  with one object exchanged; `codegen/ircross.py` runs one compiler's bitcode through another's code generator.
- **A per-object compile option changes every kernel of the object, and each batch size runs another kernel.** For
  the attention tile instances (`fattn-tile.cuh`, `launch_fattn_tile_switch_ncols1`; Qwen3.8-27B: head 256,
  `ncols2 = 2`): one query token per stream runs `<1,2>`, two `<2,2>`, three or four `<4,2>`, five to eight `<8,2>`,
  nine or more `<16,2>`. `llama-batched-bench` exercises `<16,2>` (prefill) and `<1,2>` (decode) only; MTP
  verification with `--spec-draft-n-max 3` runs `<4,2>`. Read an option on all of them before timing it:
  `codegen/tilescan.py OBJ...` (the tile object's kernels, reloads in loops / loop size) and
  `codegen/loopscan.py BUILD_A BUILD_B` (every device function of two builds; `TMPDIR` on a real disk). MTP at depth
  is timed with `mtp-depth2.sh` (five swap libraries under the installed `llama-server`, 117.7K-token prompt, 5 min
  per cell).
- **Server cells.** Send the text as a chat message (`/v1/chat/completions`): a raw `/completion` prompt of text plus
  question makes this model end at once (1 token, `stop_type eos`), which is how `mtp-depth.sh` ran 15 cells without
  a decoded token. Check `predicted_n` against what was asked in every request. When the tail of a cached prompt
  changes, the server evaluates 2,050 tokens again, not only the new ones. **A first-cell watcher waits for the TSV's
  first line or the done flag**, never for a text pattern in the progress file (the pattern missed the failure line).
- **A package is accepted on the installed deb**, at both design points, n = 5 per arm, against
  `/root/build-master-v100r` on the 10.0 runtime (`rocm-ab5.sh`, 3 h; it refuses an existing TSV), AND, when the
  change touches a tile object, at MTP depth (`mtp-accept.sh`, 45 min: the package's own GPU library against the
  previous build's). Order of a release (`-r2` is the model): build (`build-mx-llama-deb.sh SRC`),
  `deb-test-llama-101.sh OUT` (installs the deb from the file), both acceptances, `tbo-r2.sh` (the package ships no
  tests: the previous source build's `test-backend-ops` with the package's GPU library preloaded), push the fork
  commit to `master`/`gfx906-single`/`gfx906-multi` and the annotated tag, release notes and channel notes without
  placeholders, `publish-llama-r2.sh DEB` (it checks tag and master on the remote), `dogfood-llama-101.sh OUT`,
  `raw-copy-101.sh`, report, commit the tuning repository. Run `hostcc-screen.sh` or anything else that needs the
  OLD package installed before the deb test.
- **Comparing the GPU code of two builds from different source directories**: `codegen/fncmp.py` reports every
  function that references a string as different (the directory name is in the string pool, strings move, each
  PC-relative `s_add_u32` immediate changes). `codegen/maskcmp.py A.so B.so` masks those immediates and separates
  "other instructions" from "one constant". Strip the `// addr:` comments from `llvm-objdump` output before any
  textual comparison (`--no-leading-addr` keeps them), or every function differs.
- **A smoke value is not a result.** The package test's 8K first-prompt prefill read 722 tok/s for `-r2` and 749
  for `-r1` a day apart; back to back (`smoke-r2-prefill/`) they are 731 and 731.

`raw-copy-101.sh` copies the result lines into the tuning repository; `docs-update-101.py` moved its statements of the
platform state to ROCm 10.1 (run once).
