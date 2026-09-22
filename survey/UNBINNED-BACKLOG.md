# Verdicts that exist only as prose — the binning backlog

D12 says a candidate is not surveyed until a verdict record exists. By that rule the following are
**unbinned**, and they are not obscure: they are the "do not re-test" rules in `CLAUDE.md` that steer
what gets worked on. Each was measured, but only on a baseline this campaign has shown to be
misconfigured, and none has a lint-checked record.

## Two are already PROVEN WRONG, both for build-configuration reasons

| prose verdict | what is actually true |
|---|---|
| "a 4-bit V cache — measured loser or non-functional" | **+4.44% / +8.30% decode, quality-neutral.** The `q8_0-q4_0` FA kernel was simply not compiled; an uncompiled combination converts K and V to f16, with a warning in the log. `survey/q4v-cache.md` |
| "`GGML_CUDA_ALLREDUCE=internal` — a measured loser" | The internal AllReduce **requires `n_devices == 2`** (`allreduce.cu:398`). On four dies it cannot initialise, so it never ran. That is an unsupported configuration recorded as a performance result. |

## The remaining prose verdicts, all on the old baseline

`-sm layer` · `-sm row` · `GGML_CUDA_DISABLE_GRAPHS=1` · `GGML_CUDA_FORCE_CUBLAS` · the nineteen
environment knobs of run-through s.1 · the six MMVQ losers (rows 1, rows 8, rows 2/warps 2,
whole-block loads, LDS staging, aligned dword loads) · alex4300's tile rows · DPP inside the attention
kernels · six-head GQA packing · `-b 1024/512` · `-np 32` for interactive traffic · `--kv-unified` ·
"q8_0 KV is not for speed" · kernel fusion (S3).

**CORRECTED 2026-09-22:** I first wrote that these were all measured with `GGML_HIP_RCCL` OFF. That is
wrong. The `/opt` production builds (`llama.cpp-gfx906`, `llama.cpp-prod`, `llama.cpp-mxxm-fh`) all link
`librccl` with the same call sites as the current builds — they were built from the fork's own
`build.sh` recipe, which sets `GGML_HIP_RCCL=ON`. **The butterfly regression was introduced by me in the
v0.4.1 campaign**, by configuring with plain cmake defaults instead of that recipe. So the old verdicts
were measured on RCCL, and are less suspect on that axis than I claimed.
What they were measured on: **b10288/b10912 base, ROCm 7.14**, and — before 2026-09-21 — **single-user
cells at 1×32K**. `/opt/llama.cpp-prod` also lacks the `q8_0-q4_0` FA instance while
`/opt/llama.cpp-gfx906` has it, so FA coverage varied build to build. This
campaign measured baseline misconfiguration alone moving results by up to **18.7%**, which is larger
than most of the effects those verdicts turned on.

## Triage, not a blanket re-test

Re-testing all of it would cost more than the original survey. Ranked by the chance the verdict is an
artefact rather than a fact:

1. **DPP inside the attention kernels** — a "measured loser" on a build whose collective and FA kernel
   set were both wrong; DPP is also the one item the ISA review ranked last, so agreement here would be
   informative either way. One unit.
2. **The six MMVQ losers** — kernel-level verdicts that should be robust to collective choice, but they
   predate the Q8_0 repack path now confirmed at +13.6% prefill, which touches the same code.
3. **`--kv-unified` and `-np 32`** — service-shaped verdicts measured with the old harness at a
   ~33:1 prefill:decode ratio, which R2.4 says models document-QA rather than chat.
4. **`-sm layer` / `-sm row`** — `-sm row` "fails to load" is a functional claim, cheap to re-confirm.
5. **The nineteen environment knobs** — all within 0.3%; lowest priority, and a knob that did nothing
   on butterfly may do something on RCCL.

Nothing here should be cited as settled without re-measurement on the campaign configuration.
