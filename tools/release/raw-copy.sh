#!/bin/bash
# raw-copy.sh -- copy the v1.0.0 cycle's result lines (progress files and the texts the report cites) into the tuning
# repository's data/raw/2026-11-14/. Text only; no server logs, no model output beyond what the progress lines quote.
set -eu
W=/root/rebase-v100; D=/root/llama.cpp-benchmarking/data/raw/2026-11-14
mkdir -p $D
for d in pr-test-merge pr-test-master pr-test-candidates fold-master isolate-master mtp-modes-pr mtp-modes-pr-np4 mtp-modes-master mtp-rows-diag extra-pr \
         ctx-pr ctx-pr-early ctx-pr-onedie ctx-stock ctx-master deb-test deb-test/pr-test deb-test/mtp-np1 deb-test/mtp-np4 deb-test/old-np1 deb-test/old-np4 \
         dsv41-stock dsv41-pr nemotron-stock nemotron-pr nemotron-master np4-ab drafters dogfood; do
  [ -f $W/$d/progress ] || continue
  install -D -m 0644 $W/$d/progress $D/${d//\//-}.progress.txt
done
for f in queue1.progress queue2.progress pr-body.md merge-msg.txt pad-fix-msg.txt publish-100.log; do
  [ -f $W/$f ] && install -m 0644 $W/$f $D/$f
done
ls $D | wc -l
