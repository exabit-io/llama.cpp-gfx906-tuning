# Draft report for ROCm/llvm-project (not filed)

Checked 2026-11-16 07:30 UTC: ROCm/llvm-project has issues enabled; a search of its issues for `216372`, `213584`,
`getRegPressureSetLimit` and "pressure limit" returns nothing; the tip of `release/therock-10.1` is still
`d6f6cb691863` (2026-09-07). Check both again before filing. Everything below this line is the text to file.

---

Title: `release/therock-10.1` has #213584 (MachineLICM uses RegisterClassInfo limits) without its fix #216372: VGPR pressure limit wraps to ~4e9 on AMDGPU

## Summary

The compiler shipped with ROCm 10.1 (AMD clang 24.0.0git, `d6f6cb691863e9258034ab48c1dfce21689d5606`, the tip of
`release/therock-10.1`) contains

- `80df471b75d0` "Reland [MachineLICM] Use `RegisterClassInfo::getRegPressureSetLimit` (#213584)" (2026-08-03)

but not

- `1d7d16b94c61` "[CodeGen] Avoid register pressure limit underflow (#216372)" (2026-08-19),

because the last upstream commit merged into the branch is `fc636be1b4c2` (2026-08-13). `amd-staging` has the fix since
the merge of 2026-08-19 (`2053c91dae02`) and so do the `therock-10.2` and `therock-10.3` branches; `release/therock-10.1`
and `gpuep-releases/therock-10.1` (tip `22983251df75`, 2026-11-10) do not.

On AMDGPU the unsigned subtraction in `RegisterClassInfo::computePSetLimit()` wraps for the `VGPR_32` pressure set, so
MachineLICM hoists with no effective VGPR pressure limit. In the kernel below the target's limit is 128 and the value
MachineLICM receives is 4294965632 (`SReg_32`: 102 becomes 52). This is the case in all 75 functions of the translation
unit we looked at, so it is not specific to this kernel; this kernel is where it costs measurable time.

## Effect

A HIP flash-attention kernel (llama.cpp's CUDA backend built for HIP, gfx906, 128 VGPRs available at its occupancy)
compiled with `-mllvm -amdgpu-sched-strategy=max-ilp`:

| | ROCm 10.0 (clang 23, `8f497e0992fb`) | ROCm 10.1 (clang 24, `d6f6cb691863`) |
|---|---|---|
| spilled VGPRs (`.vgpr_spill_count`) | 1 | 18 |
| inner loop: instructions / scratch reloads / `s_waitcnt` | 1883 / 0 / 9 | 1910 / 11 / 14 |

Run time, Qwen3.8-27B Q8_0 on four gfx906 dies, prompt processing of one 128K-token stream, only this one object
exchanged in the library (two runs each, shuffled order; `llc` built from ROCm/llvm-project at the named commit, the same
optimized bitcode in every `llc` row):

| Object compiled by | Prompt processing, tok/s |
|---|---|
| ROCm 10.0's clang 23 | 604.1 (604.0-604.2) |
| ROCm 10.1's clang 24 | 588.7 (588.0-589.5) |
| `llc` at `2fe9fc90780d` (parent of #213584) | 600.6 (600.6-600.6) |
| `llc` at `80df471b75d0` (#213584) | 589.3 (588.6-590.0) |
| `llc` at `d6f6cb691863` (ROCm 10.1) + the `RegisterClassInfo.cpp` change of #216372 | 601.0 (600.5-601.5) |

With the whole application built by and run on each ROCm release, prompt processing is 1.4 % slower at 4 x 64K tokens
and 4.1 % slower at 1 x 255K tokens on ROCm 10.1 (five runs per arm, every 10.1 run below every 10.0 run; the 10.1
runtime under 10.0-compiled code does not show it). Token generation is unaffected, and the output is identical.

## How it was found

`git bisect` over ROCm/llvm-project between the merge base of the two compilers (`7f7028776eb2`) and ROCm 10.1's
compiler commit, 13 steps, building only `llc` and counting the scratch reloads in the kernel's inner loop: first bad
commit `80df471b75d0`. Then, on `d6f6cb691863`:

- MachineLICM reading `TRI->getRegPressureSetLimit(MF, i)` again (the one line #213584 changed): 0 reloads, the
  ROCm 10.0 loop (1883 instructions).
- the `RegisterClassInfo.cpp` change of #216372 applied: 0 reloads, the same loop.
- `-disable-machine-licm`: 0 reloads. The default scheduler instead of `max-ilp`: 0 reloads.
- upstream at `1d7d16b94c61^`: 11 reloads; at `1d7d16b94c61`: 0; `main` of 2026-11-16 (`e668aa98b09b`): 0.

## Reproducer

`tile-16x2.ll` (attached, one `amdgpu_kernel` function, 342 KB; optimized IR from ROCm 10.0's clang, readable by both
compilers):

```sh
for r in 10.0 10.1; do
  /opt/rocm/core-$r/lib/llvm/bin/clang -cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 -O3 -disable-llvm-passes \
      -mllvm -amdgpu-sched-strategy=max-ilp -S -x ir tile-16x2.ll -o tile-$r.s
  grep -E 'vgpr_spill_count|private_segment_fixed_size' tile-$r.s
done
```

ROCm 10.0: `.vgpr_spill_count: 1`, `.private_segment_fixed_size: 8`. ROCm 10.1: `.vgpr_spill_count: 18`,
`.private_segment_fixed_size: 76` (`.vgpr_count` is 128 in both).

## Request

Cherry-pick `1d7d16b94c61` (#216372) into `release/therock-10.1` and `gpuep-releases/therock-10.1` if another 10.1
compiler build is made.

## Workaround we use

`-mllvm -amdgpu-vgpr-threshold-percent=90` on the one source file (the scheduler then leaves enough registers free: no
reloads, 605.4 tok/s in the cell above against 604.5 for ROCm 10.0's object in the same session, three runs each).
