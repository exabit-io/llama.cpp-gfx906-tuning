#!/usr/bin/env python3
# docs-update-101.py -- the tuning repository's statements of the platform state, moved to ROCm 10.1 (run once, after the
# ROCm 10.1 packages are published and installed from their channels).
import re

R = '/root/llama.cpp-benchmarking/'


def sub1(path, old, new):
    s = open(R + path).read()
    assert s.count(old) == 1, (path, old[:70], s.count(old))
    open(R + path, 'w').write(s.replace(old, new))


# CLAUDE.md: the installed-packages sentence and the platform sentence
p = 'CLAUDE.md'
s = open(R + p).read()
old = ("**Installed: the deb `mx-llama-cpp-gfx906` 1.0.0 in `/opt/mx-llama.cpp` (master `df2c6e5c5`, ROCm "
       "10.0.0-gfx906+20261018172427; DT_RPATH, run its binaries directly, no `LD_LIBRARY_PATH`) from the rolling apt channel `mx-llama-cpp-gfx906` on "
       "`exabit-io/resize-amdgpu-bars`, and `python3-torch-gfx906` 2.15.0 for the system python** from the rolling apt channel `python3-torch-gfx906` there "
       "(with the `python3-typing-extensions` 4.16.0 backport it depends on).")
assert s.count(old) == 1
new = ("**Installed: the deb `mx-llama-cpp-gfx906` 1.0.0 in `/opt/mx-llama.cpp` (master `df2c6e5c5`; runtime ROCm "
       "10.1.0-gfx906+20261018172427 through its DT_RPATH, compiled with ROCm 10.0's toolchain; run its binaries directly, no `LD_LIBRARY_PATH`) from the "
       "rolling apt channel `mx-llama-cpp-gfx906` on `exabit-io/resize-amdgpu-bars`, and `python3-torch-gfx906` 2.15.0 (ROCm 10.1 build) for the system "
       "python** from the rolling apt channel `python3-torch-gfx906` there (with the `python3-typing-extensions` 4.16.0 and `python3-sympy` 1.14.0 "
       "backports it depends on). **ROCm: the 10.1 stream since 2026-11-14 (`/opt/rocm/core-10.1`, HIP 7.16, LLVM 24), with 10.0 still installed beside it "
       "because its compiler builds the llama package: LLVM 24 generates a slower head-256 flash-attention tile kernel (prefill -4.1 % at 1x255K). "
       "`reports/2026-11-14-rocm-10.1.md`, scripts and the record of the system changes in `tools/rocm101/`.**")
s = s.replace(old, new)
old = "Ubuntu 24.04, ROCm 10.0 since 2026-09-19 (7.14 purged); model Qwen3.8-27B Q8_0)."
assert s.count(old) == 1
s = s.replace(old, "Ubuntu 24.04, ROCm 10.1 since 2026-11-14 (10.0 since 2026-09-19, kept installed for its compiler; 7.14 purged); model Qwen3.8-27B Q8_0).")
# two Things-not-to-do entries that the migration adds
old = "- Do not treat the Instinct tuning guide's EPYC BIOS items as applicable; the host is an Intel Xeon W.\n"
assert s.count(old) == 1
s = s.replace(old, old +
    "- **Do not compile mx-llama.cpp with ROCm 10.1's toolchain for a build that will be measured or shipped** (2026-11-14): its LLVM 24 costs prefill\n"
    "  1.4 % at 4x64K and 4.1 % at 1x255K (decode unchanged, output identical), all of it in `fattn-tile-instance-dkq256-dv256`. Use ROCm 10.0's compiler\n"
    "  (`tools/rocm101/build-rocm.sh SRC BUILD LOG /opt/rocm/core-10.0`; the deb script's `COMPILERSTREAM=10.0`) and run on the 10.1 runtime. Already tried\n"
    "  and not a fix: `iterative-ilp` or the default scheduler for the tile kernels, tile-table nbatch_K 64, occupancy 1 (`reports/2026-11-14-rocm-10.1.md` s.4).\n"
    "  With two ROCm streams installed, every build and run names its ROCm tree: `/opt/rocm` follows the alternatives, a 10.0-toolchain build has no ROCm\n"
    "  path of its own and loads whatever `LD_LIBRARY_PATH`/`/opt/rocm/lib` gives it, and the 10.1 link driver embeds `/opt/rocm/core-10.1/lib` as DT_RPATH.\n"
    "- There is no `rocm-smi` in ROCm 10.1; `amd-smi` replaces it. Scripts set the DPM level through sysfs (`tools/gpu-test-env.sh`).\n")
open(R + p, 'w').write(s)

# ROCM-SETUP.md: header note and a dated section
p = 'ROCM-SETUP.md'
sub1(p, "> **Written for ROCm 7.14.** The box runs **ROCm 10.0** since 2026-09-19 (`/opt/rocm/core-10.0`; 7.14 purged; migration record `/root/rocm10-migration-20260919/`). The steps below are the 7.14 record.",
     "> **Written for ROCm 7.14.** The box runs **ROCm 10.1** since 2026-11-14 (`/opt/rocm/core-10.1`, beside 10.0 in `/opt/rocm/core-10.0`; record `/root/rocm101-migration-20261114/`, section at the end) and ran 10.0 from 2026-09-19 (7.14 purged; record `/root/rocm10-migration-20260919/`). The steps below are the 7.14 record.")
s = open(R + p).read().rstrip('\n')
s += '''

## 2026-11-14: ROCm 10.1 beside 10.0

The gfx906 repository's newest stream is `10.1.0-gfx906+20261018172427` (HIP 7.16.26385, LLVM 24, RCCL 2.30.7). Streams
install side by side under `/opt/rocm/core-<stream>`; `/opt/rocm/{lib,bin,include,llvm,...}` and the tools in `/usr/bin`
are alternatives that follow the newest one.

    # /etc/apt/preferences.d/rocm-gfx906:  Package: amdrocm*  /  Pin: version 10.1.0-gfx906*  /  Pin-Priority: 1001
    apt-get install $(apt-mark showmanual | grep -E '^amdrocm.*10\\.0' | sed 's/10\\.0/10.1/g')    # 64 new packages, none removed
    apt-get install amdrocm-amdsmi amdrocm-blas amdrocm-llvm amdrocm-rand amdrocm-runtime          # the unversioned names follow the pin

No reboot. 10.0 stays installed on the build host: mx-llama.cpp is compiled with its toolchain (LLVM 24 generates a slower
flash-attention tile kernel) and runs on the 10.1 runtime. A node needs only the 10.1 runtime packages, which the two
Exabit packages pull in as dependencies. ROCm 10.1 has no `rocm-smi`; use `amd-smi`. Details and measurements:
`reports/2026-11-14-rocm-10.1.md`; scripts `tools/rocm101/`.
'''
open(R + p, 'w').write(s + '\n')

# RELEASE-RUNBOOK.md: state table rows and a section on ROCm streams
p = 'RELEASE-RUNBOOK.md'
s = open(R + p).read()
m = re.search(r'^\| ROCm \| 10\.0\.0-gfx906\+20261018172427 .*$', s, flags=re.M)
assert m
s = s[:m.start()] + ("| ROCm | runtime **10.1.0-gfx906+20261018172427** (HIP 7.16.26385, LLVM 24, RCCL 2.30.7) in `/opt/rocm/core-10.1`; "
                     "10.0.0-gfx906+20261018172427 (HIP 7.15.26333, LLVM 23) stays installed in `/opt/rocm/core-10.0` as the llama package's compiler |") + s[m.end():]
m = re.search(r'^\| Installed llama\.cpp \|.*$', s, flags=re.M)
assert m
s = s[:m.start()] + ("| Installed llama.cpp | deb `mx-llama-cpp-gfx906` 1.0.0+git20261114.df2c6e5c5.rocm10.1.gfx906.20261018172427-1 (compiled by ROCm 10.0's toolchain, "
                     "runtime ROCm 10.1) from the ROLLING channel `mx-llama-cpp-gfx906` on `exabit-io/resize-amdgpu-bars` (per-version release "
                     "`mx-llama-cpp-1.0.0-gfx906-rocm10.1`; the 10.0 build of the same commit is `...-rocm10.0`) |") + s[m.end():]
m = re.search(r'^\| Installed PyTorch \|.*$', s, flags=re.M)
assert m
s = s[:m.start()] + ("| Installed PyTorch | deb `python3-torch-gfx906` 2.15.0+rocm10.1.gfx906.20261018172427-1 (torchvision 0.30.0, torchaudio 2.11.0) from the ROLLING channel "
                     "`python3-torch-gfx906`, with `python3-typing-extensions` 4.16.0-0exabit1 and `python3-sympy` 1.14.0-2~24.04.exabit1 from the same channel "
                     "(per-version release `pytorch-2.15.0-gfx906-rocm10.1`); work dir `/root/pytorch-gfx906-rocm10.1-2.15.0/` |") + s[m.end():]
old = "PyTorch traps found on 2.15.0:\n"
assert s.count(old) == 1
new = '''### A new ROCm stream (10.1, 2026-11-14)

A stream change is a platform change: nothing built by the new toolchain is published before it passes the design points.

1. Record the state, move the apt pin, install the new stream beside the old one (`ROCM-SETUP.md`, last section;
   `/root/rocm101-migration-20261114/README.md`). The alternatives switch `/opt/rocm` by themselves.
2. Build the same commit against each stream with every ROCm path explicit (`tools/rocm101/build-rocm.sh SRC BUILD LOG
   /opt/rocm/core-X`), and check which libraries each build loads (`ldd` under the run's environment): a 10.0-toolchain
   build has no ROCm path of its own, the 10.1 link driver embeds its own as `DT_RPATH`.
3. Correctness on the new stream (`suite-101.sh`), then the design-point A/B with the rule written first (`rocm-ab.sh`,
   `rocm-ab-stats.py`: n = 5 per arm per axis, 3.2 h).
4. If it regresses: `rocm-cross.sh` separates compiler from runtime (two cells per arm); `codegen/swaplink.py` + `screen.sh`
   find the object (a relinked library with one group of objects from the other compiler, loaded with `LD_PRELOAD`; 4 min
   per arm at 1 x 128K); `codegen/trycc.py` / `tileexp.py` recompile one source with other flags or table values and
   read the kernel's register use from the code object. Spilled registers were NOT a predictor of speed in the 10.1 case.
5. The packages: version `...rocm<stream>.gfx906.<stamp>` (two streams can carry one stamp); the llama deb takes
   `ROCMSTREAM` (runtime: Depends and `DT_RPATH`) and `COMPILERSTREAM` (toolchain) separately; test the package
   (`deb-test-llama-101.sh`), then accept it on the installed package with the same rule (`rocm-ab2.sh`).
   A difference between the package and a source build is located by swapping shared libraries, not by rebuilding:
   `pkg-swap.sh` preloads all of the package's libraries by full path with one group taken from the other build (6 min
   per arm at 4 x 64K; ranges of 3-4 runs separated 0.4 %). That is how the deb's host flag was found.
6. Before publishing a package that pulls the new runtime onto nodes, test the old packages on it (`suite-mixed.sh`):
   that is the state of every node between the two upgrades.

Backports of Ubuntu packages (`tools/backports/build-backport.sh DSC BUILD_DEPS`): fetch the newer series' source with
`apt-get source` through a private apt configuration (verified against the Ubuntu archive key), rebuild in the clean noble
chroot `/var/cache/exabit-backports/noble-buildd` (mmdebstrap), version `<newer>~24.04.exabit1`, publish into the channel
of the package that needs it. Check the newer package's binary relations first: Ubuntu 26.04's setuptools breaks
wheel < 0.46 and depends on eight de-vendored libraries, which is why it was not backported.

'''
s = s.replace(old, new + old)
old = "- **Compare the wheel's `Requires-Dist` with what noble ships before building the deb, and import the staged tree with the system\n"
assert s.count(old) == 1
s = s.replace(old, "- **A declared requirement that noble does not satisfy is tested, not assumed harmless** (`torch-pydeps-check.py`: export with dynamic\n"
                   "  shapes, symbolic tracing, `torch.compile` on the CPU, C++ extensions through ninja and through setuptools). sympy 1.12 broke the first\n"
                   "  three in 2.15; the first 2.15.0 notes said nothing needed the newer one and had to be corrected.\n" + old)
open(R + p, 'w').write(s)

# the morning's PyTorch report: the sympy sentence was wrong
sub1('reports/2026-11-14-pytorch-2.15.0.md',
     "Still declared and not satisfied on noble, as with 2.14.1: `sympy>=1.13.3` (1.12) and `setuptools>=77` (68). Nothing\ntested needs them.",
     "Still declared and not satisfied on noble, as with 2.14.1: `sympy>=1.13.3` (1.12) and `setuptools>=77` (68). Nothing\ntested needs them. "
     "**Corrected the same day:** the tests here did not exercise sympy; `torch.export` with dynamic shapes,\nsymbolic tracing and `torch.compile` fail with 1.12. "
     "Fixed by a `python3-sympy` 1.14.0 backport in the channel\n(`2026-11-14-rocm-10.1.md` section 6).")

# reports/README.md
sub1('reports/README.md', "| 2026-11-14 | [PyTorch 2.15.0 package, typing_extensions backport, rolling apt channel](2026-11-14-pytorch-2.15.0.md) |",
     "| 2026-11-14 | [ROCm 10.1: migration, a compiler regression in one kernel, both packages rebuilt, sympy backport](2026-11-14-rocm-10.1.md) | "
     "ROCm 10.1 installed beside 10.0; mx-llama.cpp compiled by its LLVM 24 is correct but loses prefill (-1.4 % at 4x64K, -4.1 % at 1x255K, decode unchanged): "
     "the loss is one object (head-256 flash-attention tile kernel), no scheduler or table setting restores it; the llama package "
     "for 10.1 is compiled with 10.0's toolchain and accepted (prefill = 10.0, decode -0.97 % / -0.74 %: 0.6 % the 10.1 runtime, 0.4 % the deb's "
     "`-march=native` on libllama, found by swapping libraries); PyTorch 2.15.0 on 10.1 unchanged; sympy 1.12 breaks `torch.export`/`torch.compile` (backport in the channel); "
     "setuptools not replaced |\n"
     "| 2026-11-14 | [PyTorch 2.15.0 package, typing_extensions backport, rolling apt channel](2026-11-14-pytorch-2.15.0.md) |")

# README.md: status section
p = 'README.md'
s = open(R + p).read()
old = '## Status — 2026-11-14\n'
assert s.count(old) == 1
add = ('''## Status — 2026-11-14 (evening): ROCm 10.1

The box runs the ROCm 10.1 stream (HIP 7.16, LLVM 24), installed beside 10.0. Both packages are rebuilt for it and
installed from their rolling channels. mx-llama.cpp compiled by 10.1's LLVM 24 is correct but slower at prefill (1.4 % at
4 x 64K, 4.1 % at 1 x 255K), entirely in the head-256 flash-attention tile kernel; the 10.1 runtime does not touch
prefill. The llama package for 10.1 is therefore compiled with 10.0's toolchain. It passes the design-point acceptance:
prefill equals ROCm 10.0's (805.7 against 806.0 tok/s at 4 x 64K, 410.2 against 410.2 at 1 x 255K), decode is 0.97 % and
0.74 % lower (24.546 against 24.787 tok/s per slot, 31.698 against 31.935), under the 2 % floor. Of the 4 x 64K
difference about 0.6 % is the 10.1 runtime and 0.4 % the deb's host flag (`-march=native` on `libllama`). PyTorch 2.15.0
on 10.1 is unchanged, and its channel now carries a sympy 1.14.0 backport that repairs `torch.export` and `torch.compile`
on Ubuntu 24.04. `reports/2026-11-14-rocm-10.1.md`.

''')
s = s.replace(old, add + old.replace('2026-11-14', '2026-11-14 (morning): llama.cpp v1.0.0', 1))
open(R + p, 'w').write(s)
print('docs updated for ROCm 10.1')
