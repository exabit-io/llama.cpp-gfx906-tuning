# KV cache quality gate — 2026-09-23

Perplexity for every KV type the 2026-09-23 FA sweep put in contention. Same binary
(`build-faq-allquants`, `GGML_CUDA_FA_QUANTS=all`), one flag changed, 16K context / 6 chunks of
`wiki.test.raw`, n=2 per type.

| -ctk / -ctv | PPL | +/- | vs q8_0/q8_0 |
|---|---:|---:|---:|
| f16 / f16 | 5.6264 | 0.0626 | +0.0045 |
| q8_0 / q8_0 | 5.6219 | 0.0626 | — |
| q8_0 / q4_1 | 5.6344 | 0.0627 | +0.0125 |
| q8_0 / q4_0 | 5.6216 | 0.0624 | -0.0003 |
| q4_0 / q4_0 | 5.6491 | 0.0629 | +0.0272 |

Reference cluster for v0.4.1-era upstream is ~5.62 (`CLAUDE.md`); every type lands in it.

## Reading

Every KV type in contention is quality-equivalent at this instrument's resolution, **including
`q4_0`-K** — quantising keys to 4 bits costs +0.0272 PPL, inside the +/-0.0626 error bar. That is the
surprising one, since K quantisation is where the risk is usually assumed to be.

**Do not over-read it.** The total spread across all five types is 0.0275, which is *smaller than the
error on any single estimate*. What this establishes is that none of them is grossly worse. It does not
establish equivalence. `optimize.py`'s `--max-kl 0.04` budget exists because KL divergence is the sharper
test, and before `q4_0`-K is recommended on quality grounds it needs that test rather than this one.

**Practical consequence:** quality is **not** the discriminator among these KV types. Speed and capacity
are, and the sweep settles both — f16 is fastest at every cell (+27.5% / +18.8% / +39.2% against
q8_0/q8_0) and is the configuration every launch profile fits inside the 31 GiB/die budget with.

Raw data: `data/raw/2026-09-23/kvq.tsv`. Sweep: `data/raw/2026-09-22/kvsweep.tsv`.
Scripts: `tools/campaign-2026-09/kv-quality2.sh`, `kvsweep.sh`.
