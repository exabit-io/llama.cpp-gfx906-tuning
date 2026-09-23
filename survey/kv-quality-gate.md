patch:            kv-quality-gate | -ctk/-ctv combinations | perplexity gate for every KV type in contention
axis:             multi-user
zero point:       q8_0/q8_0 at 5.6219 +/- 0.0626, same binary, one flag differs, 2026-09-23
recipe:           16K context, 6 chunks, wiki.test.raw | build-faq-allquants | RCCL + gated custom AR | n=2 per type
metric:           perplexity; a KV type is quality-acceptable if it lands inside the sampling error of q8_0/q8_0
result:           f16 5.6264 | q8_0/q8_0 5.6219 | q8_0/q4_1 5.6344 | q8_0/q4_0 5.6216 | q4_0/q4_0 5.6491 (all +/- ~0.0626)
effect:           worst case q4_0/q4_0 at +0.0272 PPL, well inside the +/-0.0626 sampling error
stats:            n=2, and perplexity is near-deterministic here (identical to 4 decimals across reps)
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral
would change if:  a KL-divergence test, which is far sharper than 16K/6 perplexity, separated them -- especially for q4_0-K, where the risk actually lives
notes:            Every KV type in contention is quality-equivalent at this instrument's resolution,
                  INCLUDING q4_0-K, which is the surprising one: quantising keys to 4 bits costs
                  +0.0272 PPL, inside the error bar.
                  DO NOT over-read this. 16K/6 perplexity is a coarse instrument with a +/-0.0626 error
                  bar, and the spread between all five types is 0.0275 -- smaller than the error on any
                  single estimate. What this establishes is that none of them is GROSSLY worse. It does
                  not establish that they are equivalent, and optimize.py's --max-kl 0.04 budget exists
                  because KL divergence is the sharper test. Before q4_0-K is recommended on quality
                  grounds it needs that test, not this one.
                  Practical consequence: quality is NOT the discriminator among these KV types. Speed and
                  capacity are, and the sweep settles those -- f16 fastest at every cell, and the only
                  type that fits every launch profile inside the 31 GiB/die budget.
