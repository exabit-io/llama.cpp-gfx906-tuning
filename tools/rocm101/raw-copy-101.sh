#!/bin/bash
# raw-copy-101.sh -- copy the ROCm 10.1 migration's result lines into the tuning repository's
# data/raw/2026-11-14/rocm101/ (the report is reports/2026-11-14-rocm-10.1.md). Text only: the scripts' progress lines, the
# per-run TSVs, the system-state listings, the publish logs and the release notes as published. No bench output, no server
# logs, no clock or SMC samples, no rocminfo (device identifiers). Safe to run again: it overwrites.
set -eu
W=/root/rocm101-migration-20261114; T=/root/pytorch-gfx906-rocm10.1-2.15.0; B=/root/backports-noble
D=/root/llama.cpp-benchmarking/data/raw/2026-11-14/rocm101
mkdir -p $D
# measurements: TSV (axis, arm, block, decode tok/s per slot, prefill tok/s) and the progress lines with the statistics
for r in rocm-ab rocm-ab2 rocm-cross screen-d128k screen-swap128k screen-tile128k pkg-swap pkg-swap2 pkg-swap3 flag-builds flag-screen; do
  for e in tsv progress; do [ -f $W/$r.$e ] && install -m 0644 $W/$r.$e $D/$r.$e; done
done
# 2026-11-15: which runtime library costs decode (rt-swap, and its perf diagnostic), the tile kernel's cause (driver test,
# cross and option arms), the GPU compile-option survey (builds, pass 1, confirmation)
for r in rt-swap rt-perf drv-test screen-drv128k tile2-build screen-tile2 gpu-opt-builds gpu-opt-screen screen-gpuopt \
         gpu-opt-confirm screen-gpuconf64 screen-gpuconf128 rocm-ab3; do
  for e in tsv progress; do [ -f $W/$r.$e ] && install -m 0644 $W/$r.$e $D/$r.$e; done
done
# 2026-11-16: the package's own decode share (pkg-resid), the survey again with the shipping compiler at both design
# points (gpu-opt24-*), the LLVM bisect of the tile kernel's cause, the ROCm 10.0-stream package's acceptance (rocm-ab4)
for r in pkg-resid pkg-resid2 gpu-opt24-builds gpu-opt24-screen screen-gpuopt24m screen-gpuopt24s gpu-opt24-confirm \
         screen-gpuconf24m screen-gpuconf24s rocm-ab4 screen-bisect screen-fixrule; do
  for e in tsv progress; do [ -f $W/$r.$e ] && install -m 0644 $W/$r.$e $D/$r.$e; done
done
# 2026-11-16, second half: the static scans (reloads inside loops, the tile object at each threshold), MTP at depth
# (two sessions), the head-256 rule's second confirmation, host compilers, the FP8 KV cache
for r in mtp-depth mtp-depth2 hostcc-builds hostcc-screen fp8-kv screen-fp8kv tile-thr-confirm screen-thr64 screen-thr128 rocm-ab5 mtp-accept; do
  for e in tsv progress; do [ -f $W/$r.$e ] && install -m 0644 $W/$r.$e $D/$r.$e; done
done
for f in loopscan-c23-c24 loopscan-c23-r1 loopscan-c24-r1 loopscan-c23-nolicm loopscan-c23-v90 loopscan-c23-v80; do
  [ -f $W/$f.txt ] && install -m 0644 $W/$f.txt $D/$f.txt
done
[ -f $W/tilethr/tilescan-thresholds.txt ] && install -m 0644 $W/tilethr/tilescan-thresholds.txt $D/tilescan-thresholds.txt
# 2026-11-17: the candidates pool's tile object read statically, the confirmation arms' tile kernels (reloads in loops,
# which arms share a kernel's instructions), output quality with an FP8 KV cache (progress lines and the KL tables)
[ -f $W/candtile/progress ] && install -m 0644 $W/candtile/progress $D/cand-tilescan.progress.txt
[ -f $W/candtile/tilescan-candidates.txt ] && install -m 0644 $W/candtile/tilescan-candidates.txt $D/tilescan-candidates.txt
for f in tilescan-o2 kcmp-tile; do [ -f $W/o2-static/$f.txt ] && install -m 0644 $W/o2-static/$f.txt $D/gpu-opt24-confirm-$f.txt; done
[ -f $W/fp8-ppl/progress ] && install -m 0644 $W/fp8-ppl/progress $D/fp8-ppl.progress.txt
for a in f16 f8 q8; do [ -f $W/fp8-ppl/$a.out ] && install -m 0644 $W/fp8-ppl/$a.out $D/fp8-ppl-$a.txt; done
[ -f $W/llvm-bisect/steps.log ] && install -m 0644 $W/llvm-bisect/steps.log $D/llvm-bisect-steps.txt
[ -f $W/llvm-bisect/bisect.log ] && install -m 0644 $W/llvm-bisect/bisect.log $D/llvm-bisect-log.txt
for f in bisect-screen fixrule-screen; do [ -f $W/llvm-bisect/$f.out ] && grep -E '^\[' $W/llvm-bisect/$f.out > $D/llvm-$f.txt; done
for d in deb-test-llama-100r1 deb-test-llama-100r1/pr-test dogfood-llama-100r1; do
  [ -f $W/$d/progress ] && install -m 0644 $W/$d/progress $D/${d//\//-}.progress.txt
done
for m in mtp-np1 mtp-np4; do [ -f $W/deb-test-llama-100r1/$m.out ] && grep -E '\[mtp-modes\]' $W/deb-test-llama-100r1/$m.out > $D/deb-test-llama-100r1-$m.progress.txt; done
[ -f $W/publish-llama-100r1.log ] && install -m 0644 $W/publish-llama-100r1.log $D/publish-llama-100r1.log
[ -f /root/mx-llama-deb/release-notes-1.0.0-rocm10.0-r1.final.md ] && install -m 0644 /root/mx-llama-deb/release-notes-1.0.0-rocm10.0-r1.final.md $D/release-notes-mx-llama-cpp-1.0.0-rocm10.0-r1.md
for a in rocr100 rt101; do [ -f $W/rt-perf/multi-$a-1.perfstat ] && install -m 0644 $W/rt-perf/multi-$a-1.perfstat $D/rt-perf-$a.perfstat.txt; done
[ -f $W/rt-perf-profile.txt ] && install -m 0644 $W/rt-perf-profile.txt $D/rt-perf-profile.txt
# correctness suites and package tests
for r in suite-101 suite-mixed queue-101 suite-t90; do [ -f $W/$r.progress ] && install -m 0644 $W/$r.progress $D/$r.progress; done
for d in pr-test-101 pr-test-mixed deb-test-llama-101 deb-test-llama-101/pr-test deb-test-llama-hybrid deb-test-llama-hybrid/pr-test dogfood-llama-101 \
         pr-test-t90 deb-test-llama-r1 deb-test-llama-r1/pr-test dogfood-llama-r1 deb-test-llama-r2 deb-test-llama-r2/pr-test dogfood-llama-r2 \
         tbo-r2; do
  [ -f $W/$d/progress ] && install -m 0644 $W/$d/progress $D/${d//\//-}.progress.txt
done
for f in fold-101.out fold-mixed.out ctx-101.out ctx-mixed.out fold-t90.out ctx-t90.out; do [ -f $W/$f ] && install -m 0644 $W/$f $D/${f%.out}.progress.txt; done
# the -r2 smoke check (package and -r1 source build back to back): the [mtp-modes] lines of each server start
for f in pkg-1 r1-1 pkg-2 r1-2; do [ -f $W/smoke-r2-prefill/$f.out ] && grep -E '\[mtp-modes\]' $W/smoke-r2-prefill/$f.out > $D/smoke-r2-prefill-$f.progress.txt; done
for d in deb-test-llama-101 deb-test-llama-hybrid deb-test-llama-r1 deb-test-llama-r2; do
  for m in mtp-np1 mtp-np4; do [ -f $W/$d/$m.out ] && grep -E '\[mtp-modes\]' $W/$d/$m.out > $D/$d-$m.progress.txt; done
done
# the system before and after
for f in packages-before.tsv manual-before.txt want-10.1.txt apt-simulate.txt alternatives-before.txt alternatives-after.txt \
         sonames-10.0.txt sonames-10.1.txt hipconfig-10.0.txt ldd-llama-before.txt publish-llama-101.log publish-torch-101.log \
         publish-llama-r1.log publish-llama-r2.log maskcmp-r1-r2.txt; do
  [ -f $W/$f ] && install -m 0644 $W/$f $D/$f
done
# PyTorch on ROCm 10.1 and the sympy backport
[ -f $T/deb-test/progress ] && install -m 0644 $T/deb-test/progress $D/torch-deb-test.progress.txt
[ -f $T/dogfood/progress ] && install -m 0644 $T/dogfood/progress $D/torch-dogfood.progress.txt
for f in new-ops.out new-stack.out new-pydeps.out ref-pydeps.out; do [ -f $T/deb-test/$f ] && install -m 0644 $T/deb-test/$f $D/torch-$f.txt; done
[ -f $B/out/sympy/sympy_1.14.0-2~24.04.exabit1_amd64.changes ] && install -m 0644 "$B/out/sympy/sympy_1.14.0-2~24.04.exabit1_amd64.changes" $D/sympy-backport.changes.txt
# the notes as published
install -m 0644 /root/mx-llama-deb/release-notes-1.0.0-rocm10.1.final.md $D/release-notes-mx-llama-cpp-1.0.0-rocm10.1.md
[ -f /root/mx-llama-deb/release-notes-1.0.0-rocm10.1-r1.final.md ] && install -m 0644 /root/mx-llama-deb/release-notes-1.0.0-rocm10.1-r1.final.md $D/release-notes-mx-llama-cpp-1.0.0-rocm10.1-r1.md
[ -f /root/mx-llama-deb/release-notes-1.0.0-rocm10.1-r2.final.md ] && install -m 0644 /root/mx-llama-deb/release-notes-1.0.0-rocm10.1-r2.final.md $D/release-notes-mx-llama-cpp-1.0.0-rocm10.1-r2.md
[ -f $T/release-notes.final.md ] && install -m 0644 $T/release-notes.final.md $D/release-notes-pytorch-2.15.0-rocm10.1.md
ls $D | wc -l
