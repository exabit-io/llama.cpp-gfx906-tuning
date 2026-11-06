#!/bin/bash
# segdl.sh URL OUT NSEG -- download URL to OUT in NSEG parallel byte ranges (resumable per segment), then join.
set -u
URL=$1; OUT=$2; N=${3:-8}
SIZE=$(curl -sIL "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')
[ -n "$SIZE" ] && [ "$SIZE" -gt 0 ] || { echo "no size for $URL"; exit 1; }
if [ -f "$OUT" ] && [ "$(stat -c %s "$OUT")" = "$SIZE" ]; then echo "$OUT complete"; exit 0; fi
SEG=$(( (SIZE + N - 1) / N ))
pids=()
for i in $(seq 0 $((N-1))); do
  s=$(( i * SEG )); e=$(( s + SEG - 1 )); [ $e -ge $SIZE ] && e=$(( SIZE - 1 ))
  want=$(( e - s + 1 )); part="$OUT.part$i"
  (
    for try in $(seq 1 500); do
      have=$( [ -f "$part" ] && stat -c %s "$part" || echo 0 )
      [ "$have" -ge "$want" ] && break
      curl -sL --fail --connect-timeout 30 --speed-limit 102400 --speed-time 60 -r $(( s + have ))-$e "$URL" >> "$part" || sleep 5   # a stalled range is abandoned and resumed
    done
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
total=0; for i in $(seq 0 $((N-1))); do total=$(( total + $(stat -c %s "$OUT.part$i") )); done
[ "$total" = "$SIZE" ] || { echo "size mismatch for $OUT: $total vs $SIZE"; exit 1; }
cat $(for i in $(seq 0 $((N-1))); do echo "$OUT.part$i"; done) > "$OUT.tmp" && mv "$OUT.tmp" "$OUT" && rm -f "$OUT".part*
echo "$OUT done ($SIZE bytes)"
