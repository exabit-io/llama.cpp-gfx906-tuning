#!/bin/bash
# raw-copy-v110.sh -- copy the v1.1.0 cycle's scripts and result lines from /root/rebase-v110 into the tuning repository
# (tools/release/v110/, data/raw/2026-11-26/v110/). Text only: progress lines, tables, the static scans, the readers'
# comparison notes, git's merge output. No server logs, no model output, no clock or SMC samples. Safe to run again.
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
ls $T | wc -l; ls $D | wc -l
