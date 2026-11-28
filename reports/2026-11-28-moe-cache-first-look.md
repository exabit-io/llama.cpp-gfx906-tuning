# 2026-11-28: upstream's MoE expert cache on gfx906, first look (experts in host RAM, hot ones in VRAM)

**Why.** The lead wants bigger MoE models (GLM, DeepSeek, Kimi) and was planning 1.5 TB of RAM per machine so that
experts can live in DRAM. This is the first look at that regime with a model already on disk. It is not a full
measurement: one run per arm, one model, and only the first table is at the service's 64K floor.

**Summary.** With every expert in RAM and a 64 GiB cache in VRAM, DeepSeek-V4.1-Flash decodes at 15.95 tok/s at 64K
on one stream (cache hit rate 96.9 %), and that placement is the only one in which it can open a 64K context at all.
Prefill is slow (38.5 tok/s at 64K, a 64K prompt in 28 minutes), but mostly not because of the RAM placement: the
cache forces layer split, which is half as fast as tensor split for this model, and this model's prefill rate falls
with depth on our kernels in any placement.

Setup: macpro2019-01, 4x gfx906, installed package `mx-llama-cpp-gfx906 1.1.0`, 125 W per die, f16 KV, 384 GB of RAM
(6 x 64 GB). Model: DeepSeek-V4.1-Flash-1017 UD-IQ1_S, 217 GiB of files, 41 layers, 110 GiB of experts. Scripts
`tools/moe-cache/` (reading rules in the headers, written before each run), raw files `data/raw/2026-11-28/moe-cache/`.
Clock columns clean.

## What the source says (no run needed)

`src/llama-moe-cache.cpp`, `docs/moe-cache.md` (in our tree since v1.0.0; flags `--moe-cache-mib`,
`--moe-cache-cpu-share`): the cache **throws under `-sm tensor`** ("MoE cache does not support tensor parallelism"),
serves ubatches of at most 32 tokens (decode and drafts; large batches upload the experts they use), caches only
layers whose experts are all in host memory, and splits its budget over the dies. Its statistics appear with `-lv 4`.

## One stream x 65536, 1,024 generated tokens (`moe-64k.sh`, one run per arm)

| Placement | Decode tok/s | Prefill tok/s | Note |
|---|---|---|---|
| experts in VRAM as far as they fit, `-sm layer` | - | - | cannot allocate a 64K context (dies at 27-30 GiB) |
| the same, `-sm tensor` | - | - | cannot allocate a 64K context (dies at 30 GiB) |
| every expert in RAM (`-cmoe`), 64 GiB cache, `-sm layer` | **15.95** | **38.5** | hit rate 96.9 %, 54.9 GiB uploaded during decode; dies at 21-22 GiB |

The arms with a 32 GiB cache, with CPU share and without a cache were not run at 64K (the run was stopped; each takes
half an hour of prefill). During the 64K prefill die 0 was 81 % busy (91 % in the last third) and the other three
16 to 24 %; the machine drew 522 W DC and the CPU zone at most 95 W.

## Diagnostics below the 64K floor (not results; 8K prompt, 64 generated tokens, one run each: `moe-upload.sh`)

| Placement | Prefill tok/s at 8K | Decode tok/s (64 tokens) |
|---|---|---|
| experts in VRAM, `-sm tensor` | 281.0 | 26.3 |
| experts in VRAM, `-sm layer` | 139.0 | 21.9 |
| every expert in RAM, 64 GiB cache, `-sm layer`, default load (mmap) | 94.1 | 6.8 (cache cold) |
| the same with `-lm none` / `-lm dio` / `-lm mlock` | 91.1 / 89.1 / 90.9 | 11.8 / 12.2 / 12.2 |

- The way the file is loaded does not change the prefill rate. An earlier reading of the 64K run ("the copy from
  host memory is the limit, about 2 GiB/s") was an inference from low power draw and is withdrawn: in layer split
  only one die works at a time, so low power says nothing, and the load mode has no effect.
- In the same split mode, RAM placement costs prefill a factor of 1.48 (139 against 94): under the 1.5 line written
  before the run, so it is not the main cause. Tensor split is twice as fast as layer split, and the cache cannot
  run in it.
- The rate falls with depth in the RAM placement: 94 tok/s over the first 8K, 38.5 over 64K. Fitted to two points
  that is about 17 s per 2048-token batch plus 1.1 s per 1K tokens of depth, some 45 times the depth cost of the dense
  Qwen3.8-27B. This model keeps a full-length KV for seven layers (head sizes 512 and 640) and a 2304-cell window
  for forty; which operation carries the depth cost, and why die 0 does most of it, is not established.

Functional check at 4K (`moe-cache-check.sh`): all three texts sensible for both models; DeepSeek-V4.1-Flash with
every expert in RAM 7.9 tok/s without the cache, 12.2 with 32 GiB, 11.0 with CPU share 0.5; Flash-Next UD-Q4_K_XL
17.8 / 16.4 / 14.5 (cache still filling in 96 tokens).

## What this says about 1.5 TB, and what it does not

- RAM is what makes room for context: with the experts in VRAM this model has no room for 64K.
- Decode with the experts in RAM is usable with a large cache (one stream; several slots not measured).
- Long prompts are slow for reasons that are in code rather than in memory: the cache's missing tensor-split
  support and this model's cost at depth. Both are patch candidates; neither is fixed by buying RAM.
- One model with an unusual attention layout is not a verdict on GLM or Kimi. Open: the remaining 64K arms, four
  slots, Flash-Next at 64K, the upstream tuner (`scripts/moe-cache-tune.py`), and a profile of the 64K prefill.
