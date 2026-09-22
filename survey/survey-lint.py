#!/usr/bin/env python3
"""Refuse incomplete survey verdicts (D12).

The >300 tok/s configuration was lost because a verdict was recorded as "regression" without
naming its axis. A rule that lives only in prose has already failed once here, so this script
is the enforcement: a verdict file missing any mandatory field, or carrying a placeholder, is
reported as INVALID and the candidate does not count as surveyed.

usage: survey-lint.py [DIR]    (default: the directory this script lives in)
exit 0 = all valid, 1 = at least one invalid
"""
import re, sys, glob, os

FIELDS = ["patch", "axis", "zero point", "recipe", "metric", "result", "effect", "stats",
          "evidence", "structural", "verdict", "bin", "would change if"]
AXES = {"multi-user", "single-user"}
VERDICTS = {"improves", "regresses", "neutral", "untested", "not-prioritised", "unresolved"}
EVIDENCE = {"confirmed-fresh", "screened-only", "group-level", "not-measurable",
            "inspection"}   # inspection = verdict from reading code/upstream state, NOT measured
BINS = {"single-user-only", "multi-user-only", "both", "regresses-both",
        "conflicts-with-another-patch",
        "neutral-drop", "neutral-required-substrate",
        "technique-requires-implementation", "upstream-already-has-it"}
# NOT a tenth outcome bin: a bookkeeping state for a candidate whose fresh confirmation is still
# running. Added 2026-09-20 because the schema allowed `verdict: unresolved` while every bin implied
# a finished outcome, so in-flight candidates could not be recorded at all -- and the plan says a
# candidate is not "surveyed" until its record exists. The invariant below keeps it honest: pending
# and unresolved imply each other, so nothing can sit in limbo with a real-looking bin.
PENDING = "unbinned-pending"

d = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
files = [f for f in sorted(glob.glob(os.path.join(d, "*.md")))
         if os.path.basename(f) not in ("TEMPLATE.md", "README.md", "UNBINNED-BACKLOG.md")]
if not files:
    print(f"no verdict files in {d} (TEMPLATE.md excluded)"); sys.exit(0)

bad = 0
for f in files:
    txt = open(f).read()
    errs = []
    vals = {}
    for k in FIELDS:
        m = re.search(rf"^{re.escape(k)}:[ \t]*(.*)$", txt, re.M)
        if not m:
            errs.append(f"missing field '{k}'"); continue
        v = m.group(1).strip()
        vals[k] = v
        if not v:
            errs.append(f"'{k}' is empty")
        elif v.startswith("<") and v.endswith(">"):
            errs.append(f"'{k}' still holds the template placeholder")
    # the two anti-recurrence checks
    if vals.get("axis") and vals["axis"] not in AXES:
        errs.append(f"axis '{vals['axis']}' is not one of {sorted(AXES)} — an axis-free or "
                    "ambiguous verdict is exactly the D4 failure")
    v = vals.get("verdict", "")
    if v and v.split(":")[0] not in VERDICTS | {"conflicts-with"}:
        errs.append(f"verdict '{v}' not recognised")
    b = vals.get("bin", "")
    if b and b not in BINS | {PENDING}:
        errs.append(f"bin '{b}' is not one of the approved bins")
    # a regression verdict with no axis-specific evidence is the classic error
    if v.startswith("regresses") and vals.get("result", "").count("/") < 2:
        errs.append("a 'regresses' verdict must carry median/p10/min on its named axis")
    # evidence level must be one of the declared kinds
    ev = vals.get("evidence", "")
    if ev and ev not in EVIDENCE:
        errs.append(f"evidence '{ev}' is not one of {sorted(EVIDENCE)}")
    # screen data can never support improves/regresses (handoff 8.2: discovery != confirmation)
    if ev == "screened-only" and v.split(":")[0] in {"improves", "regresses"}:
        errs.append("'screened-only' evidence cannot support an improves/regresses verdict — "
                    "screening is triage; confirm on FRESH runs (Part 1 s1.10)")
    # improves/regresses need BH q < 0.10 recorded
    st = vals.get("stats", "")
    if v.split(":")[0] in {"improves", "regresses"}:
        m2 = re.search(r"q=([0-9.]+)", st)
        if not m2:
            errs.append("improves/regresses requires a BH-adjusted q in 'stats' (q=<value>)")
        elif float(m2.group(1)) >= 0.10:
            errs.append(f"q={m2.group(1)} does not clear the BH FDR threshold of 0.10")
    # structural necessity is NOT a performance verdict (handoff s4.2)
    stc = vals.get("structural", "")
    if stc.startswith("required-by") and vals.get("bin") == "neutral-drop":
        errs.append("a term other patches require cannot be binned neutral-drop; "
                    "it belongs in gfx906-required on structural grounds")
    name = os.path.basename(f)
    # INVARIANT: unresolved <-> unbinned-pending. A record cannot claim a real bin while its
    # confirmation is still running, and cannot sit as pending once a verdict is in.
    _v = vals.get("verdict", "").split(":")[0]
    _b = vals.get("bin", "")
    if (_v == "unresolved") != (_b == PENDING):
        errs.append(f"verdict '{_v}' and bin '{_b}' disagree — 'unresolved' requires bin "
                    f"'{PENDING}' and vice versa")
    if errs:
        bad += 1
        print(f"INVALID  {name}")
        for e in errs: print(f"         - {e}")
    else:
        print(f"ok       {name}  [{vals['axis']}] {vals['verdict']} -> {vals['bin']}")
print(f"\n{len(files)-bad}/{len(files)} valid")
sys.exit(1 if bad else 0)
