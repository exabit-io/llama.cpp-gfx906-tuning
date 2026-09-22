patch:            exabit-terms | terms/01..29 in exabit-io/llama.cpp | GROUP verdict | our 20 applied terms on top of the substrate
axis:             multi-user
zero point:       build-substrate-v041-rccl / substrate arm measured 2026-09-22
recipe:           4x64K and 1x254K | q8_0 K | 125 W/die | --cache-ram 49152 | -ngl all | RCCL + gated custom AR
metric:           decode tok/s/slot at the design point; improves requires q<0.10 and effect >= +2%
result:           4x64K: decode +2.79%, prefill +0.10% (n.s.) (n=4 per arm)
effect:           decode +2.79%, prefill +0.10% (n.s.)
stats:            q=0.0686 where significant | BH over m=12 | n=4 fresh per arm
evidence:         group-level
structural:       standalone
verdict:          improves
bin:              multi-user-only
would change if:  per-term attribution: the group is small enough that one or two terms could carry all of it and others could regress
notes:            Sobering and worth stating plainly: our own 20 terms add +2.79% decode at the multi-user
                  design point and nothing measurable at the single-user one, against the substrate's
                  +14.23% / +24.86%. The overwhelming majority of the gain is the upstream fork's work.
                  Which of the 20 earn their place is UNRESOLVED -- they have never been tested
                  individually, only as a bundle.
