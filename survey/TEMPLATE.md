<!-- Verdict record. MANDATORY for every survey candidate (D12, lead-approved 2026-09-20).
     A candidate is NOT "surveyed" until this file exists with every field populated.
     Validate with: python3 survey/survey-lint.py
     Copy to <patch-id>.md, one file per candidate per AXIS (two files if surveyed on both). -->

patch:            <short id> | <source URL> | <commit or PR #> | <what it claims, one line>
axis:             <multi-user | single-user>
zero point:       <build id + commit> measured <YYYY-MM-DD>
recipe:           <slots x depth | KV type | W/die cap | --cache-ram | -ngl | env file>
metric:           <ONE metric line with its acceptance criterion (rule 1)>
result:           median <x> / p10 <x> / min <x> tok/s | n=<fresh confirmation runs> | spread <x>%
effect:           <+/-x.x%> vs the round's frozen base
stats:            p=<raw permutation p> q=<BH-adjusted, q<0.10 to call improves/regresses> | n=<runs>
evidence:         <confirmed-fresh | screened-only | group-level | not-measurable | inspection>
                  # inspection = the verdict comes from reading code or upstream state, not
                  # from a measurement. Never valid for improves/regresses.
structural:       <standalone | required-by:<patch-id,...>>
verdict:          <improves | regresses | neutral | untested | not-prioritised | unresolved | conflicts-with:<patch-id>>
bin:              <single-user-only | multi-user-only | both | regresses-both |
                   conflicts-with-another-patch |
                   neutral-drop | neutral-required-substrate |
                   technique-requires-implementation | upstream-already-has-it>
would change if:  <the observation that would overturn this verdict>
notes:            <free text; conflicts, build quirks, anything the next reader needs>
