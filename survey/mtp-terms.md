patch:            mtp-terms | terms/24 (17dfa2336, PR 27210 adaptive MTP draft depth) and terms/26 (c9ce0c4e0) | MTP draft-depth work
axis:             multi-user
zero point:       n/a — not built
recipe:           n/a
metric:           would be R3.9: decode with MTP on vs off, per profile
result:           not measured; the terms do not apply to v0.4.1
effect:           unknown
stats:            n/a
evidence:         inspection
structural:       standalone
verdict:          untested
bin:              technique-requires-implementation
would change if:  R3.9 being brought into scope, or upstream restoring an equivalent hook
notes:            Both depend on the fork's process_decode entry point, which upstream v0.4.1 replaced with a
                  virtual process(const llama_batch&). Carrying them forward needs that reimplemented plus
                  per-draft tensor_parallel_size. Excluded from this campaign by my scope call (stated to
                  the lead 2026-09-20, not overruled): 2 terms out of 186 would have sat on the critical
                  path. R3.9 only requires MTP measured on/off, which the existing mxxm-fh build can do
                  separately.
