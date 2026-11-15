# ROCm 10.0 -> 10.1 on macpro2019-01, 2026-11-14

Record of the migration (lead: "upgrade to the latest and rebuild"). Report for readers:
`reports/2026-11-14-rocm-10.1.md` in the tuning repository (`/root/llama.cpp-benchmarking`), which also holds a copy of this
file and of the scripts in `tools/rocm101/` and the result lines in `data/raw/2026-11-14/rocm101/`. The logs named below
are in `/root/rocm101-migration-20261114/` on the box. Previous migration: `/root/rocm10-migration-20260919/`.

## What was done to the system

| Step | How | Log |
|---|---|---|
| state before | packages, alternatives, manual marks, sonames, hipconfig, rocminfo | `packages-before.tsv`, `alternatives-before.txt`, `manual-before.txt`, `sonames-10.0.txt`, `hipconfig-10.0.txt`, `rocminfo-10.0.txt`, `preferences.d-before/` |
| apt pin | `/etc/apt/preferences.d/rocm-gfx906`: `amdrocm*` pinned to `10.1.0-gfx906*` (was `10.0.0-gfx906*`) | the file's own comment |
| install 10.1 | the 51 manually installed 10.0 package names with 10.1 in the name (`want-10.1.txt`): 64 new packages, nothing removed | `apt-simulate.txt`, `install-10.1.log` |
| unversioned names | `amdrocm-amdsmi`, `-blas`, `-llvm`, `-rand`, `-runtime` upgraded to 10.1.0 | `install-10.1-unversioned.log` |
| alternatives | switched by the packages themselves (auto mode, 10.1 has the higher priority): 26 links on 10.1; `rocm-smi` stays on 10.0 because 10.1 has none | `alternatives-after.txt` |
| test environment | `/root/rocm-tests/bench/gpu-test-env.sh`: `rocm-smi --setperflevel` replaced by a sysfs write (backup `.pre-rocm101`) | |
| build chroot | `mmdebstrap` installed; noble chroot `/var/cache/exabit-backports/noble-buildd` (for the sympy backport) | `install-mmdebstrap.log`, `/root/backports-noble/` |

ROCm 10.0 is still installed (`/opt/rocm/core-10.0`): it holds the compiler the llama package is built with.
Rollback of the stream: set the pin back to `10.0.0-gfx906*`, `apt-get install` the five unversioned names at 10.0.0, and
`update-alternatives --set` the links (or remove the 10.1 packages; the alternatives fall back by themselves).

## What ROCm 10.1 is

`10.1.0-gfx906+20261018172427` (the same build stamp as the 10.0 stream's current build): HIP 7.16.26385 (10.0: 7.15.26333),
AMD clang 24.0.0git (10.0: 23.0.0git), RCCL 2.30.7 (10.0: 2.30.4), amd_smi 27.1.0. Library sonames are those of 10.0, minus
`liboam.so.1` and `librocm_smi64.so.1` (no `rocm-smi`; `amd-smi` replaces it), plus four new ones (`sonames-10.1.txt`).

## Measurements

| Script | What | Result |
|---|---|---|
| `suite-101.sh` | correctness of master df2c6e5c5 compiled by and run on 10.1 | all as on 10.0 |
| `rocm-ab.sh`, `rocm-ab-stats.py` | 10.0 against 10.1 at 4x64K and 1x255K, n=5, rule fixed before data | decode unchanged; prefill -1.39 % and -4.13 %: 10.1-compiled build REJECTED |
| `rocm-cross.sh` | compiler or runtime? the two crossed arms | the prefill loss is the compiler's; the 10.1 runtime leaves prefill alone (its decode cost: `pkg-swap.sh` below) |
| `suite-mixed.sh` | correctness of 10.0-compiled code on the 10.1 runtime | all as on 10.0 |
| `codegen/` (`spilldiff.py`, `metacmp.py`, `trycc.py`, `tileexp.py`, `swaplink.py`), `screen.sh` | which kernel, and can a setting fix it | one object: the head-256 flash-attention tile kernel; no scheduler strategy or table setting tried restores it |
| `deb-test-llama-101.sh`, `rocm-ab2.sh` | the package as published (10.0 compiler, 10.1 runtime), n=5 per arm | ACCEPTED: prefill = 10.0 (805.7 / 410.2 tok/s), decode -0.97 % at 4x64K and -0.74 % at 1x255K, under the 2 % floor |
| `publish-llama-101.sh`, `dogfood-llama-101.sh` | release `mx-llama-cpp-1.0.0-gfx906-rocm10.1`, rolling channel, reinstall from the channel | published 2026-11-15 00:12 UTC; chain verified |
| `pkg-swap.sh` (three sessions) | where the package's decode difference is: its libraries swapped for a source build's by `LD_PRELOAD` | 0.6 % the ROCm 10.1 runtime, 0.4 % the deb's `-march=native` on `libllama`/`libllama-common` |
| `flag-builds.sh`, `flag-screen.sh` | other host compiler flags (lead's question): skylake-avx512, x86-64-v4, `-mtune=intel`, no vectorizer, `-O2`, no semantic interposition, LTO, PGO | n = 3 per arm, 33 cells, none discarded: no arm beats no flags (24.672); `-march=native` -0.41 %, the 512-bit arms -0.62 and -0.77 %, `-O2` -1.29 %; vectorizer off = no flags; LTO and PGO gain nothing (report, section 5.2) |
| `publish-torch-101.sh`, `/root/pytorch-gfx906-rocm10.1-2.15.0/`, `/root/backports-noble/` | PyTorch 2.15.0 for ROCm 10.1 and the `python3-sympy` 1.14.0 backport | published 2026-11-14 into the channel `python3-torch-gfx906` |

Builds: `/root/build-master-v100r` (10.0, the reference of every comparison), `/root/build-master-v101` (10.1),
`/root/build-master-v100rn` (reference + `-march=native` host flags), `/root/build-flag-*` (host-flag variants). The
cross-over builds, the scheduler variants of the tile kernels and the object-swap libraries were scratch and are
rebuilt by their scripts (`rocm-cross.sh`, `screen.sh`, `codegen/swaplink.py`).

`raw-copy-101.sh` copies the result lines into the tuning repository; `docs-update-101.py` moved its statements of the
platform state to ROCm 10.1 (run once).
