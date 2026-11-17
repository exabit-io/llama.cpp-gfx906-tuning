# data/raw/2026-11-14/rocm101 - ROCm 10.1 migration

Result lines of the runs behind `reports/2026-11-14-rocm-10.1.md`, copied from `/root/rocm101-migration-20261114/` on
macpro2019-01 by `tools/rocm101/raw-copy-101.sh`. The scripts are in `tools/rocm101/`.

| Files | What |
|---|---|
| `rocm-ab.*` | ROCm 10.0 against 10.1 (compiled and run), 4 x 64K and 1 x 255K, n = 5 per arm; the rule is in the script header |
| `rocm-cross.*` | the two crossed arms (compiler of one stream, runtime of the other), one run each |
| `screen-d128k.*`, `screen-swap128k.*`, `screen-tile128k.*` | 1 x 128K screens: depth, object groups swapped between compilers, scheduler and table variants of the tile kernel |
| `rocm-ab2.*` | acceptance of the published package against the ROCm 10.0 source build, same rule |
| `pkg-swap.*`, `pkg-swap2.*`, `pkg-swap3.*` | library-swap sessions that locate the package's decode difference (runtime, host flag) |
| `flag-builds.progress`, `flag-screen.*` | host compiler flag variants and their screen |
| `rt-swap.*`, `rt-perf.*`, `rt-perf-*.perfstat.txt`, `rt-perf-profile.txt` | 2026-11-15: which library of the 10.1 runtime costs decode (one group at a time taken from 10.0, n = 3), and the perf diagnostic of the two ROCr libraries |
| `drv-test.progress`, `screen-drv128k.*`, `tile2-build.progress`, `screen-tile2.*` | 2026-11-15: the tile kernel's cause: driver pipeline with and without LTO, bitcode crossed between the compilers' code generators, single options on the one object |
| `gpu-opt-builds.progress`, `gpu-opt-screen.progress`, `screen-gpuopt.*` | 2026-11-15: survey of GPU-side compile options: the builds with their code comparison, and pass 1 (one 4 x 64K sample per variant, three of the reference) |
| `gpu-opt-confirm.progress`, `screen-gpuconf64.*`, `screen-gpuconf128.*` | 2026-11-15: the package's compiler: reference, plain 10.1 build, 10.1 with the rule; n = 3 shuffled at 4 x 64K and 1 x 128K |
| `rocm-ab3.*` | acceptance of the `-r1` package (master 290fc642a compiled by 10.1) against the ROCm 10.0 source build, same rule, n = 5 per arm |
| `pkg-resid.*` | 2026-11-16: is any of the `-r1` package's 4 x 64K decode difference its own? reference on 10.0, the package's program with the reference's libraries on 10.1, the package; n = 5 |
| `llvm-bisect-steps.txt`, `llvm-bisect-log.txt` | 2026-11-16: bisect of ROCm/llvm-project for the tile kernel's spills: one line per `llc` built (commit, date, verdict, hot loop: instructions, global loads, reloads, spills, waits, LDS, moves), the endpoint, option, revert and fix readings, and `git bisect log` |
| `screen-bisect.*`, `llvm-bisect-screen.txt` | 2026-11-16: the bisect's objects timed at 1 x 128K (controls, parent, first bad commit, one-line revert, upstream's fix), n = 2 shuffled |
| `screen-fixrule.*`, `llvm-fixrule-screen.txt` | 2026-11-16: the tree's rule on a compiler with the fix: clang 23, fix, fix + threshold 90, threshold 90 without the fix; 1 x 128K, n = 3 shuffled |
| `gpu-opt24-builds.progress`, `gpu-opt24-screen.progress`, `screen-gpuopt24m.*`, `screen-gpuopt24s.*` | 2026-11-16: the compile-option survey again with ROCm 10.1's compiler: builds with their code comparison, pass M (4 x 64K) and pass S (1 x 255K), one sample per variant, three of the reference per pass |
| `gpu-opt24-confirm.progress`, `screen-gpuconf24m.*`, `screen-gpuconf24s.*`, `gpu-opt24-confirm-tilescan-o2.txt`, `gpu-opt24-confirm-kcmp-tile.txt` | 2026-11-17: confirmation pass of that survey's two "up" variants against the shipped rule: threshold 90, threshold 85, `-O2` on each of the two trees, threshold 80 on every source; 4 x 64K and 1 x 255K, n = 3 shuffled; and the five arms' head-256 tile kernels read statically (reloads in loops; which arms share a kernel's instructions) |
| `cand-tilescan.progress.txt`, `tilescan-candidates.txt` | 2026-11-17: the candidates pool's head-256 tile source compiled six ways (clang 23 and 24, with and without `max-ilp` and threshold 85): reloads in loops per kernel, static |
| `fp8-ppl.progress.txt`, `fp8-ppl-f16.txt`, `fp8-ppl-f8.txt`, `fp8-ppl-q8.txt` | 2026-11-17: output quality with an FP8 KV cache on the installed `-r2` package: perplexity 16K x 6 with the f16 cache, and the KL-divergence tables of the FP8 and q8_0 caches against its logits. In the progress lines of the two KL runs the first value is the labelled difference `PPL(Q)-PPL(base)`, not a perplexity: the perplexities are in the table files |
| `rocm-ab4.*`, `deb-test-llama-100r1*`, `dogfood-llama-100r1*`, `publish-llama-100r1.log` | 2026-11-16: the ROCm 10.0-stream package rebuilt from 290fc642a without the global host flag: package test, acceptance against the ROCm 10.0 source build on the same runtime (n = 5 per arm), publish log, install by version from the channel |
| `loopscan-*.txt`, `tilescan-thresholds.txt` | 2026-11-16: static scans: scratch reloads inside loops in every device function of two builds (clang 23, clang 24, the `-r1` code, no machine LICM, the threshold on every source), and the head-256 tile object's 34 kernels at every VGPR threshold (reloads in loops / loop size) |
| `mtp-depth.*`, `mtp-depth2.*` | 2026-11-16: MTP decode at 117.7K depth (`--spec-draft-n-max 3`), five tile objects under the installed `llama-server`, n = 3 shuffled; `mtp-depth` is the first session, whose raw prompt ended at one token (no decode data) |
| `tile-thr-confirm.progress`, `screen-thr64.*`, `screen-thr128.*` | 2026-11-16: thresholds 90, 85 and 88 against the reference at the design-point kernels, 4 x 64K and 1 x 128K, n = 3 shuffled |
| `hostcc-builds.progress`, `hostcc-screen.*` | 2026-11-16: the host libraries built by gcc 13 (with and without OpenMP), Ubuntu clang 19 and ROCm 10.1's clang 24, 4 x 64K, n = 3 shuffled |
| `fp8-kv.progress`, `screen-fp8kv.*` | 2026-11-16: an FP8 KV cache at 4 x 64K: clang 23 against the `-r1` GPU code, and FP8 against f16, n = 3 shuffled |
| `mtp-accept.*`, `rocm-ab5.*`, `tbo-r2.progress.txt`, `maskcmp-r1-r2.txt`, `smoke-r2-prefill-*`, `deb-test-llama-r2*`, `dogfood-llama-r2*`, `publish-llama-r2.log` | 2026-11-16: package `-r2` (master 224b73e24, threshold 85): MTP acceptance at depth (the package's GPU library, `-r1`'s, the clang 23 tile object; n = 3), design-point acceptance against the ROCm 10.0 source build (n = 5 per arm), `test-backend-ops` on the package's GPU library, its GPU code against `-r1`'s, the back-to-back smoke check, package test, channel install, publish log |
| `suite-*.progress`, `pr-test-*`, `fold-*`, `ctx-*`, `deb-test-llama-*`, `dogfood-llama-*` | correctness suites and package tests: `101` = compiled and run on 10.1, `mixed` = 10.0-compiled on the 10.1 runtime, `hybrid` = the first published package, `t90` = master 290fc642a compiled by 10.1, `r1` = the `-r1` package |
| `torch-*`, `sympy-backport.changes.txt` | PyTorch 2.15.0 on ROCm 10.1: package test, channel install, operator, stack and Python-dependency checks (`ref` = with Ubuntu's sympy 1.12) |
| `packages-before.tsv`, `manual-before.txt`, `want-10.1.txt`, `apt-simulate.txt`, `alternatives-*.txt`, `sonames-*.txt`, `hipconfig-10.0.txt`, `ldd-llama-before.txt` | the system before and after |
| `publish-*.log`, `release-notes-*.md` | the publish logs and the release notes as published |

TSV columns: axis, arm, block, decode tok/s per slot, prefill tok/s. The 8K and 16K values in the suite and package-test
files are correctness smoke values (R2.2), never serving results; the 1 x 128K screens are single runs used to rank
variants, not results. In the `screen-*.tsv` files the first column is the cell and the third the block.
