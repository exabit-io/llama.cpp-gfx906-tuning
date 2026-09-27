patch:            substrate-62d4be47d | https://github.com/mxxm-t/mx-llama.cpp | 62d4be47d | meta: gather an axis-0 source for ops that read whole rows (drafter / DFlash under -sm tensor)
axis:             multi-user
zero point:       master 82868aa3b, read 2026-09-27 (round 2 per-group check, 0 GPU)
recipe:           this instrument: Qwen3.8-27B Q8_0, -sm tensor across four dies (tps 4, one stage), q8_repack on, no drafter
metric:           none — classified by reading the commit and its own measurements, not measured here
result:           not measured: inactive here: "A graph that never reads whole rows from a tensor split along axis 0 plans no gather and builds exactly as before"; it fixes an abort with a drafter (MTP/DFlash, R3.9) that this instrument does not run
effect:           none measurable on this instrument
stats:            n/a (not measured) | n=0
evidence:         not-measurable
structural:       standalone
verdict:          untested
bin:              neutral-required-substrate
would change if:  MTP or DFlash drafting is benchmarked under -sm tensor (R3.9); it is what makes that configuration run at all
notes:            Later mxxm-t commit listed in r2-preregistration.md as "classified by inspection". Applies equally to the
                  single-user axis (same one-stage tensor split). Stays in master with the substrate.
