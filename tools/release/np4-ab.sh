#!/bin/bash
# np4-ab.sh OUT -- is the single-stream MTP decode at -np 4 slower on v1.0.0 master than on v0.9.0 master?
# The package test saw 75.6-76.7 (1.0.0) against 78.1-78.5 tok/s (0.9.0), one server each. Here: source builds of the two
# masters, alternated over three rounds, a new server per arm per round, three short greedy requests per server.
#   v090 host      the only row path v0.9.0 has
#   v100 host      LLAMA_MTP_DEV=0, the same row path on v1.0.0
#   v100 combined  the v1.0.0 default (device rows at -np 4)
# Waits for queue2 (the GPUs are its until then). Smoke depth (8K per slot): build to build only.
set -u
W=/root/rebase-v100; O=$1; mkdir -p $O; P=$O/progress
OLD=/root/build-master-v090; NEW=/root/build-master-v100
log(){ echo "$(date -Is) [np4-ab] $*" | tee -a $P; }
until grep -q 'QUEUE2 DONE' $W/queue2.progress 2>/dev/null; do sleep 15; done
[ -x $OLD/bin/llama-server ] && [ -x $NEW/bin/llama-server ] || { log "a build is missing"; exit 1; }
vals(){ v=$(grep 'run [123]' $1/progress | sed 's/.*gen 256 tok //; s/ tok\/s.*//' | tr '\n' ' '); echo "${v:-none}"; }
for r in 1 2 3; do
  ARMS="host" bash $W/mtp-modes.sh $OLD $O/v090-host-$r 4 > $O/v090-host-$r.out 2>&1; log "round $r v090 host rc=$?: $(vals $O/v090-host-$r)"
  ARMS="host" bash $W/mtp-modes.sh $NEW $O/v100-host-$r 4 > $O/v100-host-$r.out 2>&1; log "round $r v100 host rc=$?: $(vals $O/v100-host-$r)"
  ARMS="combined" bash $W/mtp-modes.sh $NEW $O/v100-combined-$r 4 > $O/v100-combined-$r.out 2>&1; log "round $r v100 combined rc=$?: $(vals $O/v100-combined-$r)"
done
python3 -I - $O <<'EOF' | tee -a $P
import glob, re, statistics, sys
for arm in ("v090-host", "v100-host", "v100-combined"):
    xs = []
    for f in sorted(glob.glob("%s/%s-[123]/progress" % (sys.argv[1], arm))):
        xs += [float(m.group(1)) for m in re.finditer(r"run [123]: .*gen 256 tok ([0-9.]+) tok/s", open(f).read())]
    if xs:
        print("[np4-ab] %-14s n=%d mean %.2f min %.2f max %.2f sd %.2f tok/s" % (arm, len(xs), statistics.mean(xs), min(xs), max(xs), statistics.pstdev(xs)))
    else:
        print("[np4-ab] %-14s no samples" % arm)
EOF
log "=== NP4-AB DONE ==="
