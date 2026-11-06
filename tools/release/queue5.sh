#!/bin/bash
# queue5.sh -- after queue3: (1) RCCL topology A/B with the 2026-09-04 rccl-tests invocation: generated file via
# /etc/rccl.conf (no env), the firmware's links, no topology; (2) llama.cpp master RCCL init from /etc/rccl.conf;
# (3) the mx-llama-cpp-gfx906 0.9.0 deb: build from a clean worktree of gfx906/v0.9.0/r0/master, install, test.
set -u
W=/root/rebase-v090
Q=$W/queue5.progress
log(){ echo "$(date -Is) [queue5] $*" | tee -a $Q; }
until grep -q 'QUEUE3 DONE\|not running' $W/queue3.progress 2>/dev/null; do sleep 20; done
log "start"
. /root/rocm-tests/bench/gpu-test-env.sh
O=$W/rccl-topo-ab; mkdir -p $O
/usr/local/sbin/rccl-topology show --links firmware > $O/firmware.xml 2> /dev/null
setmax
cd /root/rccl-tests/build
for arm in generated firmware none; do
  case $arm in
    generated) e="";;                                        # /etc/rccl.conf -> /var/cache/rccl/topology.xml
    firmware)  e="NCCL_TOPO_FILE=$O/firmware.xml";;
    none)      e="NCCL_CONF_FILE=/dev/null";;                # skip /etc/rccl.conf: RCCL's own detection
  esac
  env $e NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH ./all_reduce_perf -b 1M -e 1G -f 2 -g 4 -n 20 -w 5 -c 1 > $O/$arm.txt 2>&1
  log "rccl $arm: rc=$? 1GiB busbw $(awk '$1==1073741824 {print $8" / "$12}' $O/$arm.txt) GB/s, avg $(awk -F: '/Avg bus bandwidth/ {print $2}' $O/$arm.txt | tr -d ' '); topo: $(grep -m1 -oE 'NCCL_TOPO_FILE set by [a-z ]+ to [^ ]+' $O/$arm.txt || echo 'no topology file')"
done
restore
cd $W
LDP=/root/build-master-v090/bin:/opt/rocm/lib
LD_LIBRARY_PATH=$LDP HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO timeout 300 /root/build-master-v090/bin/llama-completion -m /root/models/Qwen3.5-0.8B-Q8_0.gguf \
  --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -c 1024 -n 8 --temp 0 -no-cnv --no-warmup -p "Hello" > $O/llama-init.out 2> $O/llama-init.log < /dev/null
log "llama.cpp master RCCL init via /etc/rccl.conf: rc=$? failures: $(grep -ciE 'NCCL init failed|probe AllReduce failed' $O/llama-init.log); topo: $(grep -m1 -oE 'NCCL_TOPO_FILE set by [a-z ]+ to [^ ]+' $O/llama-init.log)"

# the deb: host-CPU heavy, nothing else runs now
S=/root/wt-deb-v090
[ -d $S ] || git -C /root/exabit-llama.cpp worktree add -q --detach $S gfx906/v0.9.0/r0/master
git -C $S checkout -q --detach gfx906/v0.9.0/r0/master
bash /root/mx-llama-deb/build-mx-llama-deb.sh $S > $W/deb-build.out 2>&1; log "deb build rc=$? $(tail -1 /root/mx-llama-deb/build.log | cut -c1-200)"
DEB=$(cat /root/mx-llama-deb/last-deb.txt)
apt-get install -y --allow-downgrades "$DEB" > $W/deb-install.out 2>&1; log "deb install rc=$? installed $(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906)"
log "isolation: $(LD_LIBRARY_PATH=/opt/llama.cpp/lib ldd /opt/mx-llama.cpp/bin/llama-server | grep -E 'libggml|libllama' | awk '{print $3}' | sort -u | tr '\n' ' ')"
SRC=$S SKIP_TBO=1 bash $W/pr-test.sh /opt/mx-llama.cpp $W/pr-test-deb > $W/pr-test-deb.out 2>&1; log "pr-test deb rc=$?"
bash $W/mtp-ab.sh /opt/mx-llama.cpp $W/mtp-deb > $W/mtp-deb.out 2>&1; log "mtp deb rc=$?"
log "=== QUEUE5 DONE ==="
