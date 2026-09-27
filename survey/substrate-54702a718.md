patch:            substrate-54702a718 | https://github.com/mxxm-t/mx-llama.cpp | 54702a718 | cuda: retry a failed buffer allocation after releasing the repack upload scratch
axis:             multi-user
zero point:       master 82868aa3b, read 2026-09-27 (round 2 per-group check, 0 GPU)
recipe:           this instrument: Qwen3.8-27B Q8_0, -sm tensor across four dies (tps 4, one stage), q8_repack on, no drafter
metric:           none — classified by reading the commit and its own measurements, not measured here
result:           not measured: inactive here: acts only after an out-of-memory allocation failure, which the 27B at 4x64K / 1x255K never hits
effect:           none measurable on this instrument
stats:            n/a (not measured) | n=0
evidence:         not-measurable
structural:       standalone
verdict:          untested
bin:              neutral-required-substrate
would change if:  a layout close to the memory limit (large repacked expert tensors); robustness, not speed
notes:            Later mxxm-t commit listed in r2-preregistration.md as "classified by inspection". Applies equally to the
                  single-user axis (same one-stage tensor split). Stays in master with the substrate.
