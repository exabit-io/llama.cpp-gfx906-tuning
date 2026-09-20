#!/bin/bash
# scriptcheck-selftest.sh — asserts scriptcheck.sh against planted fixtures.
#
# WHY: on 2026-09-20 a "fix" to scriptcheck.sh moved its finding-append into a pipeline subshell,
# so every append was discarded and it reported "no FATAL findings" on a file with a planted bug.
# Reading the patch did not reveal that. Only running it against a known-bad file did. So the
# checker's own correctness is asserted here, in BOTH directions:
#   BAD  fixture -> the check must fire exactly N times
#   GOOD fixture -> the same check must fire ZERO times (no crying wolf; a checker that flags
#                   correct code gets ignored, which defeats it as surely as missing the bug)
# Every fixture is the REAL form that bit, copied in shape from the script it bit in.
set -u
SC="$(dirname "$0")/scriptcheck.sh"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
pass=0; fail=0

# check <name> <id> <expected-count> <<fixture
chk() {
  local name="$1" id="$2" want="$3"
  local file="$W/$name.sh"
  cat > "$file"
  local got
  # An id like T4b counts check firings; a SEVERITY (FATAL/WARN/ACK) counts lines of that severity.
  case "$id" in
    FATAL|WARN|ACK) got=$("$SC" "$file" 2>/dev/null | grep -cE "^[[:space:]]+$id[[:space:]]" || true);;
    *)              got=$("$SC" "$file" 2>/dev/null | grep -c "$id:" || true);;
  esac
  if [ "$got" = "$want" ]; then
    pass=$((pass+1)); printf '  ok    %-28s %s fired %s (want %s)\n' "$name" "$id" "$got" "$want"
  else
    fail=$((fail+1)); printf '  FAIL  %-28s %s fired %s (want %s)\n' "$name" "$id" "$got" "$want"
    "$SC" "$file" 2>&1 | sed 's/^/          /'
  fi
}

echo "=== T10  one 'local' that assigns and references (fanpower.sh, variance-gate.sh)"
chk t10-bad      T10 1 <<'EOF'
#!/bin/bash
sample() { local tag=$1 secs=$2 f=/tmp/x-$tag.log; : > $f; }
EOF
chk t10-good     T10 0 <<'EOF'
#!/bin/bash
sample() { local tag=$1; local secs=$2; local f=/tmp/x-$tag.log; : > $f; }
EOF
chk t10-bad-inbrace T10 1 <<'EOF'
#!/bin/bash
f() { local a=$1 out=/tmp/$a; }
EOF
chk t10-good-multiline T10 0 <<'EOF'
#!/bin/bash
f() {
  local a=$1
  local out=/tmp/$a
}
EOF

echo "=== T7  kill with a :-0 default signals the whole process group"
chk t7-bad       T7 1 <<'EOF'
#!/bin/bash
kill ${SRV:-0} 2>/dev/null
EOF
chk t7-good      T7 0 <<'EOF'
#!/bin/bash
kpid() { [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] && kill "$1" 2>/dev/null; }
kpid "${SRV:-}"
EOF

echo "=== T4  unanchored p*grep -f matches the tool's own shell (exit 144)"
chk t4-bad       T4 1 <<'EOF'
#!/bin/bash
pkill -f 'chain-noqueue|chain-final'
EOF
chk t4-good      T4 0 <<'EOF'
#!/bin/bash
pkill -f '^/bin/bash /root/night-20260919/chain-final.sh'
EOF

echo "=== T4b wait loop keyed on a pattern (chain-gate.sh never ran)"
chk t4b-bad      T4b 1 <<'EOF'
#!/bin/bash
while pgrep -f 'test-backend-ops' >/dev/null; do sleep 30; done
EOF
chk t4b-good     T4b 0 <<'EOF'
#!/bin/bash
while kill -0 "$PREV_PID" 2>/dev/null; do sleep 30; done
EOF

echo "=== T4b anchored wait loop: WARN, not FATAL (chain-build.sh, chain-close.sh)"
chk t4b-anchored FATAL 0 <<'EOF'
#!/bin/bash
while pgrep -f "^/bin/bash /root/night-20260919/serve[.]sh" >/dev/null 2>&1; do sleep 30; done
EOF

echo "=== comments are prose, not code (run-gates.sh:3 was flagged for describing the trap)"
chk comment-only FATAL 0 <<'EOF'
#!/bin/bash
# T4 variant learned 2026-09-20: a wait loop keyed on `pgrep -f <pattern>` can match
# this shell. Also never `kill ${x:-0}`, and never `pkill -x llama-batched-bench`.
echo ok
EOF

echo "=== T4c p*kill -x with comm over 15 chars matches nothing (ladder.sh)"
chk t4c-bad      T4c 1 <<'EOF'
#!/bin/bash
pkill -x llama-batched-bench
EOF
chk t4c-good     T4c 0 <<'EOF'
#!/bin/bash
pkill -x llama-server
EOF

echo "=== T3  trap handler that never exits: bash clears traps and RESUMES"
chk t3-bad       T3 1 <<'EOF'
#!/bin/bash
cleanup() { echo stopping; pkill -x llama-server; }
trap cleanup INT TERM
EOF
chk t3-good      T3 0 <<'EOF'
#!/bin/bash
cleanup() { echo stopping; pkill -x llama-server; exit 130; }
trap cleanup INT TERM
EOF

echo "=== T5b metric scraped with no empty check (reported '0.0 W')"
chk t5b-bad      T5b 1 <<'EOF'
#!/bin/bash
w=$(grep PZ0G /tmp/smc.log | awk '{print $2}')
echo "power $w W"
EOF
chk t5b-good     T5b 0 <<'EOF'
#!/bin/bash
w=$(grep PZ0G /tmp/smc.log | awk '{print $2}')
[ -z "$w" ] && { echo "FATAL: no PZ0G samples" >&2; exit 1; }
echo "power $w W"
EOF

echo "=== explicit waivers: named + reasoned -> ACK, not FATAL; bare/mismatched -> still FATAL"
chk waiver-honoured FATAL 0 <<'EOF'
#!/bin/bash
# scriptcheck: ok T4 — pattern is validated as anchored at runtime by the case guard below
pkill -f "$pat"
EOF
chk waiver-wrong-id FATAL 1 <<'EOF'
#!/bin/bash
# scriptcheck: ok T7 — waiver names a different check, so T4 must still fire
pkill -f "$pat"
EOF
chk waiver-acked    ACK 1 <<'EOF'
#!/bin/bash
# scriptcheck: ok T4 — reason recorded here
pkill -f "$pat"
EOF

echo "=== a waiver may cover SEVERAL checks as a list, and still only the listed ones"
chk waiver-list-both FATAL 0 <<'EOF'
#!/bin/bash
# scriptcheck: ok T4,T4b — pattern is refused unless anchored, and the loop is bounded by WAIT_MAX
while pgrep -f "$pat" >/dev/null 2>&1; do sleep 15; done
EOF
chk waiver-list-partial T4b 1 <<'EOF'
#!/bin/bash
# scriptcheck: ok T4 — only T4 is waived here, so the T4b finding must survive
while pgrep -f "$pat" >/dev/null 2>&1; do sleep 15; done
EOF

echo "=== meta: a file with ALL bad fixtures must exit 1; an all-good file must exit 0"
cat > "$W/allbad.sh" <<'EOF'
#!/bin/bash
sample() { local tag=$1 f=/tmp/x-$tag.log; }
kill ${SRV:-0}
pkill -f 'chain-final'
pkill -x llama-batched-bench
EOF
if "$SC" "$W/allbad.sh" >/dev/null 2>&1; then
  fail=$((fail+1)); echo "  FAIL  all-bad file exited 0 (this is the 2026-09-20 subshell defect)"
else pass=$((pass+1)); echo "  ok    all-bad file exits 1"; fi
cat > "$W/allgood.sh" <<'EOF'
#!/bin/bash
sample() { local tag=$1; local f=/tmp/x-$tag.log; }
kpid() { [ "${1:-0}" -gt 1 ] && kill "$1"; }
pkill -f '^/bin/bash /root/x.sh'
pkill -x llama-server
EOF
if "$SC" "$W/allgood.sh" >/dev/null 2>&1; then
  pass=$((pass+1)); echo "  ok    all-good file exits 0"
else fail=$((fail+1)); echo "  FAIL  all-good file exited 1 — the checker cries wolf"; "$SC" "$W/allgood.sh" | sed 's/^/          /'; fi

echo
echo "scriptcheck self-test: $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
