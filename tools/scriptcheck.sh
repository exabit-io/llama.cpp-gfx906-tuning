#!/bin/bash
# scriptcheck.sh — mechanical checks for the bug classes that actually bit on 2026-09-19/20.
#
# WHY THIS EXISTS: every one of those bugs READ exactly like what was intended. Reviewing them by
# reading could not catch them, because reading is the faculty that produced them. So each is encoded
# here as a grep that does not care what the author meant.
#
# STRUCTURAL RULE (learned the hard way on 2026-09-20): every finding is appended by add() IN THE
# CURRENT SHELL. Never `cmd | while read ...; do findings+=(); done` — the loop body of a pipeline
# runs in a subshell and every append is silently discarded, which is how the first version of this
# file reported "no FATAL findings" on a file with a planted bug. Feed loops with <<< or < <().
#
#   usage: scriptcheck.sh FILE...        exit 1 if any FATAL finding
#          scriptcheck.sh --self-test    verify the checks against planted fixtures
set -u

if [ "${1:-}" = "--self-test" ]; then exec "$(dirname "$0")/scriptcheck-selftest.sh"; fi

# Emit the file as `LINENO:text` with COMMENT-ONLY lines blanked (never removed: keeping the line
# preserves numbering). A comment that merely DESCRIBES a bug is not the bug.
# Blank COMMENT-ONLY lines IN PLACE — same line count, so the `grep -n` downstream still reports
# true line numbers. (First attempt numbered the lines here as well; the `^`-anchored checks then
# saw "12:w=$(grep ...)" and silently matched nothing. The self-test caught it, reading did not.)
nocomment() { sed -E 's/^[[:space:]]*#.*$//' "$1"; }

fatal=0
for f in "$@"; do
  [ -f "$f" ] || continue
  findings=()
  # An explicit, VISIBLE waiver: `# scriptcheck: ok T4b — <reason>` on the offending line (or the one
  # above it) downgrades that check to ACK and prints the reason. This exists because a few forms are
  # correct in ways a grep cannot see — e.g. waitproc.sh passes a VARIABLE to pgrep -f and validates
  # that it is anchored at runtime. A waiver must name the check and give a reason, and it still shows
  # up in the audit, so silently ignoring a finding is never the path of least resistance.
  waived() { # waived LINE ID  -> prints the reason, or nothing
    # A waiver may cover several checks: `# scriptcheck: ok T4,T4b — reason`. The id must appear in
    # the comma/space list BEFORE the reason text, so a waiver naming T7 never silences T4.
    local l="$1" id="$2" txt w ids
    txt=$(sed -n "$((l>2?l-2:1)),${l}p" "$f" 2>/dev/null)
    while IFS= read -r w; do
      [ -z "$w" ] && continue
      ids=$(printf '%s' "$w" | sed -E 's/^scriptcheck: ok[[:space:]]+//; s/[[:space:]]*[-—:].*$//')
      printf '%s' "$ids" | tr ',' ' ' | grep -qwF "$id" || continue
      printf '%s\n' "$w"; return 0
    done <<< "$(printf '%s\n' "$txt" | grep -oE "scriptcheck: ok[[:space:]]+[A-Za-z0-9,[:space:]]+[-—:][^\"']*" || true)"
    return 0
  }
  add() { # add SEVERITY LINE ID MESSAGE
    local sev="$1" ln="$2" id="$3" msg="$4" w=""
    [ "$ln" != "-" ] && w=$(waived "$ln" "$id")
    if [ -n "$w" ]; then
      # strip the leading "scriptcheck: ok <ID>" so only the human reason remains
      local why; why=$(printf '%s' "$w" | sed -E "s/^scriptcheck: ok[[:space:]]+[A-Za-z0-9,[:space:]]+[[:space:]]*[-—:][[:space:]]*//")
      findings+=("ACK    $ln  $id: waived — ${why:-no reason given}"); return
    fi
    findings+=("$sev  $ln  $id: $msg"); [ "$sev" = "FATAL" ] && fatal=1; return 0
  }

  # ---- T10: one `local`/`declare`/`export` that assigns a var AND references it.
  # bash expands every word BEFORE any assignment takes effect, so the reference is EMPTY.
  # Judged PER STATEMENT, not per line: `local a=$1; local b=$a` is correct and must not flag,
  # while `local a=$1 b=$a` is the bug. Splitting on ';' is what distinguishes them.
  while IFS=: read -r ln txt; do
    [ -z "${ln:-}" ] && continue
    while IFS= read -r stmt; do
      [ -z "$stmt" ] && continue
      vars=$(printf '%s\n' "$stmt" | grep -oE '[a-zA-Z_][a-zA-Z0-9_]*=' | tr -d '=')
      for v in $vars; do
        if printf '%s\n' "$stmt" | grep -qE "\\\$\{?$v\b"; then
          add "FATAL" "$ln" "T10" "one 'local' assigns \$$v AND references it — bash expands all words before assigning, so it is EMPTY here. Use separate 'local' statements."
        fi
      done
    done <<< "$(printf '%s\n' "$txt" | tr ';' '\n' | grep -E '(^|[[:space:]])(local|declare|export)[[:space:]]' || true)"
  done < <(nocomment "$f" | grep -nE '(^|[{;&|][[:space:]]*)(local|declare|export)[[:space:]]+[a-zA-Z_]+=.*[[:space:]][a-zA-Z_]+=')

  # ---- T7: kill with a :-0 default -> `kill 0` signals the ENTIRE PROCESS GROUP, incl. this script
  while IFS=: read -r ln _; do
    [ -z "${ln:-}" ] && continue
    add "FATAL" "$ln" "T7" "kill \${x:-0} — 'kill 0' signals the ENTIRE PROCESS GROUP, killing the script itself. Guard with a kpid() helper."
  done <<< "$(nocomment "$f" | grep -nE 'kill[[:space:]]+(-[0-9A-Za-z]+[[:space:]]+)?"?\$\{[a-zA-Z_]+:-0\}"?' || true)"

  # ---- T4: pkill/pgrep -f whose pattern is not anchored to ^ or an absolute path
  while IFS=: read -r ln txt; do
    [ -z "${ln:-}" ] && continue
    printf '%s\n' "$txt" | grep -qE "p(kill|grep)[^|;]*-f[[:space:]]+['\"]?\^" && continue
    printf '%s\n' "$txt" | grep -qE "p(kill|grep)[^|;]*-f[[:space:]]+['\"]?/" && continue
    add "FATAL" "$ln" "T4" "unanchored 'p*grep -f' matches ANY shell quoting the pattern, including this tool's own. Anchor to ^/abs/path, or use a PID."
  done <<< "$(nocomment "$f" | grep -nE 'p(kill|grep)[^|;]*-f[[:space:]]' || true)"

  # ---- T4b: a WAIT LOOP keyed on a pattern — blocks forever when the monitor matches itself
  while IFS=: read -r ln txt; do
    [ -z "${ln:-}" ] && continue
    # An ANCHORED wait pattern (^/bin/bash /abs/path) cannot match this tool's own shell, whose
    # cmdline is `/bin/bash -c ...`. That is a WARN (pattern-keyed chaining is still fragile), not
    # the FATAL that actually hung chain-gate.sh on an unanchored 'test-backend-ops'.
    if printf '%s\n' "$txt" | grep -qE "p(grep|kill)[^|;]*-f[[:space:]]+[\"']?(\^|/)"; then
      add "WARN" "$ln" "T4b" "wait loop keyed on a pattern. Anchored, so it cannot match this shell — but chaining by SEQUENCE or PID is still the rule."
    else
      add "FATAL" "$ln" "T4b" "wait loop keyed on an UNANCHORED pattern — it matches the monitoring shell and never exits. This is what hung chain-gate.sh. Chain by SEQUENCE or PID."
    fi
  done <<< "$(nocomment "$f" | grep -nE 'while[[:space:]]+.*p(grep|kill)[^;]*-f' || true)"

  # ---- T4c: -x with a comm longer than 15 chars silently matches NOTHING
  while IFS=: read -r ln txt; do
    [ -z "${ln:-}" ] && continue
    for nm in $(printf '%s\n' "$txt" | grep -oE '\-x[[:space:]]+[A-Za-z0-9_.-]+' | awk '{print $2}'); do
      [ ${#nm} -gt 15 ] && add "FATAL" "$ln" "T4c" "'p*kill -x $nm' — comm is truncated to 15 chars (${#nm} given), so this matches NOTHING."
    done
  done <<< "$(nocomment "$f" | grep -nE 'p(kill|grep)[^|;]*-x[[:space:]]' || true)"

  # ---- T3: a trap handler that never exits -> bash clears traps and RESUMES the script
  if grep -qE '^[[:space:]]*(cleanup|on_exit|trap_handler)\(\)' "$f" 2>/dev/null; then
    awk '/^[[:space:]]*(cleanup|on_exit|trap_handler)\(\)/{f=1} f&&/exit[[:space:]]/{ok=1} f&&/^\}/{f=0} END{exit ok?0:1}' "$f" \
      || add "FATAL" "-" "T3" "trap handler never calls 'exit' — bash clears traps and RESUMES the script, re-running work with guards removed."
  fi

  # ---- T5b: a metric scraped from text into a var that is printed with no empty check
  while IFS=: read -r ln txt; do
    [ -z "${ln:-}" ] && continue
    v=$(printf '%s\n' "$txt" | grep -oE '^[[:space:]]*[a-zA-Z_][a-zA-Z0-9_]*=' | tr -d ' =')
    [ -z "${v:-}" ] && continue
    grep -qE "(\[ -z \"?\\\$\{?$v|\[\[ -z|: \\\$\{$v:\?)" "$f" || \
      add "WARN" "$ln" "T5b" "'$v' is scraped from text with no empty-check — an empty metric gets reported as a result (this is what produced '0.0 W')."
  done <<< "$(nocomment "$f" | grep -nE '^[[:space:]]*[a-zA-Z_][a-zA-Z0-9_]*=\$\((grep|awk|sed)' || true)"

  if [ ${#findings[@]} -gt 0 ]; then
    echo "=== $f"
    printf '  %s\n' "${findings[@]}"
  fi
done
[ $fatal -eq 0 ] && echo "scriptcheck: no FATAL findings" || echo "scriptcheck: FATAL findings present"
exit $fatal
