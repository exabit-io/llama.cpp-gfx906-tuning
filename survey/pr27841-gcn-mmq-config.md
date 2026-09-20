patch:            pr27841-gcn-mmq-config | https://github.com/ggml-org/llama.cpp/pull/27841 | PR 27841 (MERGED 2026-09-12) | per-arch MMQ config for AMD GCN: wave64 nthreads 512 instead of falling back to the RDNA2 wave32/256 config; tile widths to J=128
axis:             multi-user
zero point:       not yet measured — Stage C rebase will absorb this patch into stock master
recipe:           4x64K | q8_0 KV | 125 W/die | --cache-ram 49152 | -ngl all | settings/gfx906.env
metric:           prefill t/s at 64K depth, 4 slots; adopted if >= zero point + noise band
result:           median n/a / p10 n/a / min n/a tok/s | n=0 | spread n/a
effect:           n/a — not measured
stats:            n/a — no confirmation runs collected
evidence:         inspection
structural:       standalone
verdict:          untested
bin:              upstream-already-has-it
would change if:  the rebase shows our local copy diverges from the merged version, or the merged
                  version is gated to an arch predicate that excludes gfx906 — then it returns to
                  the survey as a normal candidate and needs a real A/B.
notes:            This is the project's "S1" patch. We carried it locally; upstream merged it on
                  2026-09-12. Concrete instance of D3's rationale: surveying before the rebase would
                  have spent an A/B measuring something upstream already has. Verify after C2 whether
                  our commit is now redundant, and whether the merged form keeps the J=128 tile width
                  (our tile-table work depends on it).
