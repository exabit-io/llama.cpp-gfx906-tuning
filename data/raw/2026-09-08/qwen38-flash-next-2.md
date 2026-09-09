# qwen38-flash-next-2 2026-09-08T22:10:00+00:00: /opt/llama.cpp-gfx906-master-r2, four dies -sm tensor, LLAMA_PLE_SHARD=1 (--load-mode dio)

## per-sample tg128 x8 (llama-bench -p 0 -n 128 -r 8), repack on / off
| run | mean ± sd | samples |
|---|---:|---|
| repack on | 36.4 ± 9.9 | 24.3 17.0 42.0 41.9 41.6 41.6 41.6 41.6 |
| repack off | 37.2 ± 8.7 | 26.7 20.0 42.1 41.9 41.7 41.7 41.6 41.6 |

## perplexity -c 2048 --chunks 8: wikitext-2 test vs held-out corpus (our own 2026-09 docs, 250915 bytes)
| model | wikitext | held-out |
|---|---:|---:|
| Flash-Next UD-Q4_K_XL (branch, PLE sharded) | 2.0044 +/- 0.03653 | 9.4273 +/- 0.27828 |
| 27B Q8_0 (production) | 5.0722 +/- 0.12973 | 8.8344 +/- 0.24600 |
# done 2026-09-08T22:21:02+00:00
