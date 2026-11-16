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
| `rocm-ab4.*`, `deb-test-llama-100r1*`, `dogfood-llama-100r1*`, `publish-llama-100r1.log` | 2026-11-16: the ROCm 10.0-stream package rebuilt from 290fc642a without the global host flag: package test, acceptance against the ROCm 10.0 source build on the same runtime (n = 5 per arm), publish log, install by version from the channel |
| `suite-*.progress`, `pr-test-*`, `fold-*`, `ctx-*`, `deb-test-llama-*`, `dogfood-llama-*` | correctness suites and package tests: `101` = compiled and run on 10.1, `mixed` = 10.0-compiled on the 10.1 runtime, `hybrid` = the first published package, `t90` = master 290fc642a compiled by 10.1, `r1` = the `-r1` package |
| `torch-*`, `sympy-backport.changes.txt` | PyTorch 2.15.0 on ROCm 10.1: package test, channel install, operator, stack and Python-dependency checks (`ref` = with Ubuntu's sympy 1.12) |
| `packages-before.tsv`, `manual-before.txt`, `want-10.1.txt`, `apt-simulate.txt`, `alternatives-*.txt`, `sonames-*.txt`, `hipconfig-10.0.txt`, `ldd-llama-before.txt` | the system before and after |
| `publish-*.log`, `release-notes-*.md` | the publish logs and the release notes as published |

TSV columns: axis, arm, block, decode tok/s per slot, prefill tok/s. The 8K and 16K values in the suite and package-test
files are correctness smoke values (R2.2), never serving results; the 1 x 128K screens are single runs used to rank
variants, not results. In the `screen-*.tsv` files the first column is the cell and the third the block.
