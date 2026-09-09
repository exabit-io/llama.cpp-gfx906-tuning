#!/bin/bash
# Download unsloth/Qwen3.8-Flash-Next-GGUF UD-Q4_K_XL (4 shards, ~104 GiB) + the MTP draft head (Q8_0), resumable, two streams at a time.
D=/root/models/Qwen3.8-Flash-Next-UD-Q4_K_XL; B=https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main; L=/root/rocm-tests/bench/dl-flash-next.log
cd $D || exit 1
log() { echo "$(date -Is) $*" >> $L; }
get() { local f=$1; local n=$(basename $f); log "start $n"; curl -sL -C - --retry 20 --retry-delay 10 -o "$n" "$B/$f" && log "done $n $(stat -c %s "$n") bytes" || log "FAILED $n rc=$?"; }
log "START"
get UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
get UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf &
get UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf &
wait
get UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf &
get MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf &
wait
log "ALLDONE $(du -sh $D | cut -f1)"; touch $D/.download-done
