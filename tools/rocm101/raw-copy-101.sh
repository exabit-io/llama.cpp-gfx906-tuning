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
for a in rocr100 rt101; do [ -f $W/rt-perf/multi-$a-1.perfstat ] && install -m 0644 $W/rt-perf/multi-$a-1.perfstat $D/rt-perf-$a.perfstat.txt; done
[ -f $W/rt-perf-profile.txt ] && install -m 0644 $W/rt-perf-profile.txt $D/rt-perf-profile.txt
# correctness suites and package tests
for r in suite-101 suite-mixed queue-101 suite-t90; do [ -f $W/$r.progress ] && install -m 0644 $W/$r.progress $D/$r.progress; done
for d in pr-test-101 pr-test-mixed deb-test-llama-101 deb-test-llama-101/pr-test deb-test-llama-hybrid deb-test-llama-hybrid/pr-test dogfood-llama-101 \
         pr-test-t90 deb-test-llama-r1 deb-test-llama-r1/pr-test dogfood-llama-r1; do
  [ -f $W/$d/progress ] && install -m 0644 $W/$d/progress $D/${d//\//-}.progress.txt
done
for f in fold-101.out fold-mixed.out ctx-101.out ctx-mixed.out fold-t90.out ctx-t90.out; do [ -f $W/$f ] && install -m 0644 $W/$f $D/${f%.out}.progress.txt; done
for d in deb-test-llama-101 deb-test-llama-hybrid deb-test-llama-r1; do
  for m in mtp-np1 mtp-np4; do [ -f $W/$d/$m.out ] && grep -E '\[mtp-modes\]' $W/$d/$m.out > $D/$d-$m.progress.txt; done
done
# the system before and after
for f in packages-before.tsv manual-before.txt want-10.1.txt apt-simulate.txt alternatives-before.txt alternatives-after.txt \
         sonames-10.0.txt sonames-10.1.txt hipconfig-10.0.txt ldd-llama-before.txt publish-llama-101.log publish-torch-101.log \
         publish-llama-r1.log; do
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
[ -f $T/release-notes.final.md ] && install -m 0644 $T/release-notes.final.md $D/release-notes-pytorch-2.15.0-rocm10.1.md
ls $D | wc -l
