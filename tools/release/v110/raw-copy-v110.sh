#!/bin/bash
# raw-copy-v110.sh -- copy the v1.1.0 cycle's scripts and result lines from /root/rebase-v110 into the tuning repository
# (tools/release/v110/, data/raw/2026-11-26/v110/). Text only: progress lines, tables, the static scans, the readers'
# comparison notes, git's merge output, the final static scans, the package test's progress files, the tag messages.
# No server logs, no model output, no clock or SMC samples. Safe to run again.
set -eu
W=/root/rebase-v110; R=/root/llama.cpp-benchmarking; T=$R/tools/release/v110; D=$R/data/raw/2026-11-26/v110
mkdir -p $T/tile $D/cmp $D/tile
for f in $W/*.sh $W/*.py; do install -m 0755 $f $T/$(basename $f); done
for f in $W/tile/*.sh; do install -m 0755 $f $T/tile/$(basename $f); done
install -m 0644 $W/README.md $T/README.md
install -m 0644 $W/merge-msg.txt $W/pr-body.md $T/
for f in $W/cmp/*.md; do install -m 0644 $f $D/cmp/$(basename $f); done
for f in $W/*.progress $W/merge-v110.merge.out; do [ -f $f ] && install -m 0644 $f $D/$(basename $f); done
for d in $W/*/; do n=$(basename $d); [ -f $d/progress ] && install -m 0644 $d/progress $D/$n.progress.txt; done
for f in $W/screen-*.tsv $W/*.tsv; do [ -f $f ] && install -m 0644 $f $D/$(basename $f); done
for f in $W/tile/tilescan-*.txt; do [ -f $f ] && install -m 0644 $f $D/tile/$(basename $f); done
for f in $W/final-static/*.txt; do [ -f $f ] && install -m 0644 $f $D/$(basename $f); done
for d in $W/deb-test-v110/*/; do n=$(basename $d); [ -f $d/progress ] && install -m 0644 $d/progress $D/deb-test-v110-$n.progress.txt; done
for d in $W/*/; do n=$(basename $d); [ -f $d/requests.tsv ] && install -m 0644 $d/requests.tsv $D/$n.requests.tsv; done
for f in $W/tagmsg-*.txt; do [ -f $f ] && install -m 0644 $f $T/$(basename $f); done
# 2026-11-28: the eviction logs, the kernel trace, the push and publish outputs, the comment drafted for PR #25
for f in $W/evsample-*.log $W/push-v110.out $W/publish-llama-110.out; do [ -f $f ] && install -m 0644 $f $D/$(basename $f).txt; done
[ -f $W/kfdtrace-v110/trace.txt ] && install -m 0644 $W/kfdtrace-v110/trace.txt $D/kfdtrace-v110.trace.txt
[ -f $W/kfdtrace-v110/ev.log ] && install -m 0644 $W/kfdtrace-v110/ev.log $D/kfdtrace-v110.ev.log.txt
[ -f $W/pr25-rates-comment.md ] && install -m 0644 $W/pr25-rates-comment.md $T/pr25-rates-comment.md
ls $T | wc -l; ls $D | wc -l
