#!/bin/bash
# queue-clamped.sh -- CORRECTNESS ONLY, with the dies latched at 1000 MHz (2026-11-26 evening): the final verification
# (queue-final.sh with the fold check) of master-v110 and of the candidates pool, with the watchdog's clock guard off
# (SCLK_GUARD=0; its DC envelope guard stays). Pass/fail, perplexities, texts and draft counts do not depend on the
# clock; NO tok/s of these runs is a number (tags end in "c" for clamped). Nothing else runs on the host meanwhile.
# Expected: test-backend-ops 21871/21871 on each die; master PPL 16K/6 5.6088 (the pre-final build's); the fold check's
# on/off pair as its header says (they agree to rounding, only the GDN L2 fold is not bit-exact); test-llama-context
# 148/150 with the two KV-replica models in both rollback modes; mtp-carry the same counts for every request A;
# rs-modes3's four rules.
set -u
W=/root/rebase-v110; P=$W/queue-clamped.progress
log(){ echo "$(date -Is) [queue-clamped] $*" | tee -a $P; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is running"; exit 1; }
log "clock now: $(grep -h '\*' /sys/class/drm/card[0-9]/device/pp_dpm_sclk | tr -s ' ' | tr '\n' ' ')"
SCLK_GUARD=0 FOLD=1 bash $W/queue-final.sh /root/build-master-v110 masterc /root/wt-master-v110 > $W/queue-final-masterc.out 2>&1; log "master rc=$?"
SCLK_GUARD=0 FOLD=1 bash $W/queue-final.sh /root/build-candidates-v110 candc /root/wt-candidates-v110 > $W/queue-final-candc.out 2>&1; log "candidates rc=$?"
log "=== QUEUE DONE ==="
