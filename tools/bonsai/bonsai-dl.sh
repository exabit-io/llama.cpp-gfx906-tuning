#!/bin/bash
# bonsai-dl.sh -- download Ternary-Bonsai-2.5-27B-Q2_0.gguf (prism-ml/Ternary-Bonsai-2.5-27B-GGUF, public, Apache-2.0)
# at the repository revision read on 2026-11-28 and check it against the SHA256 the hub lists for the file.
# No credentials are sent (the repository is not gated); the client names itself "curl".
set -u
REV=28a7093cc25bc151cc2b284c5570f12ce343fdcf
F=Ternary-Bonsai-2.5-27B-Q2_0.gguf; DST=/root/models/$F; W=/root/bonsai-bench
WANT=$(python3 -I -c '
import json,sys
d=json.load(open(sys.argv[1]))
for s in d["siblings"]:
    if s["rfilename"]==sys.argv[2]: print(s["lfs"]["sha256"], s["size"])' $W/hf/api.json $F)
SHA=${WANT% *}; SIZE=${WANT#* }
echo "$(date -Is) want $F: $SIZE bytes, sha256 $SHA"
curl -fL -sS -A curl -C - --retry 5 --retry-delay 5 -o "$DST.part" "https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF/resolve/$REV/$F" -w "$(date -Is) http %{http_code}, %{size_download} bytes in %{time_total} s, %{speed_download} B/s\n"
rc=$?
got=$(stat -c %s "$DST.part" 2>/dev/null)
echo "$(date -Is) curl rc=$rc, file $got bytes"
[ "$got" = "$SIZE" ] || { echo "SIZE MISMATCH"; exit 1; }
have=$(sha256sum "$DST.part" | cut -d' ' -f1)
[ "$have" = "$SHA" ] || { echo "SHA256 MISMATCH: $have"; exit 1; }
mv "$DST.part" "$DST"; echo "$(date -Is) verified: $DST"
