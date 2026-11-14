#!/usr/bin/env python3
# docs-update.py -- the tuning repo's statements of the code state, moved to the v1.0.0 cycle (run once, at the end).
import re

R = '/root/llama.cpp-benchmarking/'


def sub1(path, old, new):
    s = open(path).read()
    assert s.count(old) == 1, (path, old[:70], s.count(old))
    open(path, 'w').write(s.replace(old, new))


# CLAUDE.md: the code paragraph (one line)
p = R + 'CLAUDE.md'
s = open(p).read()
m = re.search(r'^\*\*Code: one repository, .*$', s, flags=re.M)
assert m
new = ("**Code: one repository, [`exabit-io/mx-llama.cpp`](https://github.com/exabit-io/mx-llama.cpp). Since 2026-11-14 the substrate** is "
       "`merge-v1.0.0` (`a73c601e6`) = mxxm-t's fork at `cdc81dae4` + llama.cpp v1.0.0 (`b1b770d`, b12359) + two HIP fixes (grouped top-k ballot, "
       "`PAD_REFLECT_1D` launch size), offered to mxxm-t as [PR #24](https://github.com/mxxm-t/mx-llama.cpp/pull/24) (#17, #22 and #23 were closed unmerged). "
       "Branches: `master` = `gfx906-single` = `gfx906-multi` = `df2c6e5c5` (substrate + RCCL default + both-profile patches), `gfx906-candidates` (unbinned), "
       "`merge-v1.0.0` (pure substrate). **Taking the next upstream release through the stack: `RELEASE-RUNBOOK.md`** (procedure, scripts in `tools/release/`, "
       "and the silent-break classes to check). `exabit-io/llama.cpp` is retired history. mixa3607's `ML-gfx906` is a different project (ROCm builds + Docker "
       "presets, no llama.cpp source). **Installed: the deb `mx-llama-cpp-gfx906` 1.0.0 in `/opt/mx-llama.cpp` (master `df2c6e5c5`, ROCm "
       "10.0.0-gfx906+20261018172427; DT_RPATH, run its binaries directly, no `LD_LIBRARY_PATH`) from the rolling apt channel `mx-llama-cpp-gfx906` on "
       "`exabit-io/resize-amdgpu-bars`, and `python3-torch-gfx906` 2.14.1 for the system python** from its signed release there. Plan: `RE-RE-SURVEY-ACTION-PLAN.md`.")
s = s[:m.start()] + new + s[m.end():]
open(p, 'w').write(s)

# README.md: the code paragraph and a new status section
p = R + 'README.md'
s = open(p).read()
m = re.search(r'^The code is a separate repository: .*$', s, flags=re.M)
assert m
new = ("The code is a separate repository: **https://github.com/exabit-io/mx-llama.cpp**. **The substrate since 2026-11-14** is `merge-v1.0.0` (`a73c601e6`) = "
       "mxxm-t's fork at `cdc81dae4` merged with llama.cpp v1.0.0 (`b1b770d`, b12359) plus two HIP fixes, offered to mxxm-t as "
       "[PR #24](https://github.com/mxxm-t/mx-llama.cpp/pull/24) (PR #17, #22 and #23, the v0.5.0, v0.7.0 and v0.9.0 merges, were closed unmerged). Branches: "
       "`master` = `gfx906-single` = `gfx906-multi` = `df2c6e5c5` (substrate + RCCL default + the patches binned for both profiles), `gfx906-candidates` (our "
       "unbinned patches), `merge-v1.0.0` (the pure substrate, until merged). How a release is taken through the stack: `RELEASE-RUNBOOK.md`. "
       "`exabit-io/llama.cpp` is retired history (archived 2026-09-24, read-only). mixa3607's `ML-gfx906` is a different project (ROCm builds + Docker presets, "
       "no llama.cpp source). Plan: `RE-RE-SURVEY-ACTION-PLAN.md`.")
s = s[:m.start()] + new + s[m.end():]
old = '## Status — 2026-10-22\n'
assert s.count(old) == 1
status = ('''## Status — 2026-11-14

llama.cpp v1.0.0: merged into mxxm-t's master and offered as PR #24 (replacing #23), the 18 Exabit commits rebased onto it without a port. Perplexity and
every fold-check value are unchanged from v0.9.0 (16K/6 5.6093 on `master`); `test-backend-ops` passes 20388/20388 where stock v1.0.0 aborts on gfx906
(`reports/2026-11-14-v100-rebase.md`). The build ships as `mx-llama-cpp-gfx906` 1.0.0 from the rolling apt channel `mx-llama-cpp-gfx906` on
`exabit-io/resize-amdgpu-bars` (one source line; `apt upgrade` brings later versions) and is installed on the box from there. ROCm is still the 10.0 build
20261018172427. The v0.9.0 cycle of 2026-11-06 (PR #23, `gfx906-rccl-topology`) is in `reports/2026-11-06-v090-rebase.md`.

''')
s = s.replace(old, status + old)
open(p, 'w').write(s)

# REQUIREMENTS.md: change log, factual pointer only
sub1(R + 'REQUIREMENTS.md', '## Change log\n\n',
     '## Change log\n\n- 2026-11-14 (lead\'s instruction, recorded by Claude; no requirement changed): **R3.7 substrate moved** to `merge-v1.0.0` (mxxm-t master '
     '`cdc81dae4` + llama.cpp v1.0.0 + two HIP fixes), offered as mxxm-t/mx-llama.cpp#24; #23 (v0.9.0, substrate from 2026-11-06) was closed unmerged. '
     'Packages now publish into a rolling apt channel per package.\n')

# RE-RE-SURVEY-ACTION-PLAN.md, both copies: a dated note at the top of section 2
for p in (R + 'RE-RE-SURVEY-ACTION-PLAN.md', '/root/RE-RE-SURVEY-ACTION-PLAN.md'):
    old = '> **2026-10-22 — substrate moved, the v0.5.0 text below is history.**'
    note = ('> **2026-11-14 — substrate moved again (v0.9.0 on 2026-11-06, v1.0.0 today).** Substrate = `merge-v1.0.0` (`a73c601e6`) = mxxm-t master\n'
            '> `cdc81dae4` + llama.cpp v1.0.0 + two HIP fixes, offered as [mxxm-t/mx-llama.cpp#24](https://github.com/mxxm-t/mx-llama.cpp/pull/24)\n'
            '> (#22 and #23 closed unmerged). `master` = `gfx906-single` = `gfx906-multi` = `df2c6e5c5` (the same 18 Exabit commits; PPL 16K/6 5.6093,\n'
            '> unchanged from v0.9.0); `gfx906-candidates` = `5c01f4ffd`. Tags `gfx906/v1.0.0/r0/*`, `gfx906/v0.9.0/r0/*`. Installed build: deb\n'
            '> `mx-llama-cpp-gfx906` 1.0.0 from the rolling channel. Details: memory `mx-llama-cpp-v100-pr`, work dir /root/rebase-v100,\n'
            '> `reports/2026-11-14-v100-rebase.md`. The 2026-10-22 note below is history.\n>\n')
    sub1(p, old, note + old)

# reports/README.md: the index row and the header line
p = R + 'reports/README.md'
sub1(p, '`master` = substrate `merge-v0.7.0` + both-profile patches', '`master` = substrate `merge-v1.0.0` + both-profile patches')
old = '| 2026-11-06 | [v0.9.0 rebase, PR #23, package, generated RCCL topology](2026-11-06-v090-rebase.md) |'
row = ('| 2026-11-14 | [v1.0.0 rebase, PR #24, package, rolling apt channel](2026-11-14-v100-rebase.md) | merge + rebase + deb 1.0.0 in the new channel '
       '`mx-llama-cpp-gfx906`; perplexity and folds unchanged from v0.9.0; MTP device rows vs host rows (identical output); two upstream HIP bugs '
       '(top-k ballot, `PAD_REFLECT_1D` launch size); the fork\'s KV-replica state-copy failure on four dies |\n')
sub1(p, old, row + old)
print('docs updated')
