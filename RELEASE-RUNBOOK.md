# Release runbook: taking a llama.cpp release through the gfx906 stack

Written 2026-10-22, after the v0.7.0 cycle (reconcile with mxxm-t, merge, rebase, PR, packages). Follow it in order for the
next upstream release. Every script named here is in `tools/release/` (box-side copies; the working copies and their logs
from the v0.7.0 cycle are in `/root/rebase-v070/` on macpro2019-01). Report: `reports/2026-10-22-v070-rebase.md`.

## 0. State after the v1.0.0 cycle (2026-11-14)

| What | Where / value |
|---|---|
| Code repository | `github.com/exabit-io/mx-llama.cpp`; local clone `/root/exabit-llama.cpp`, remote `exabit-mx`; upstream fork remote `mxxm` (mxxm-t/mx-llama.cpp), `upstream` (ggml-org) |
| Pure substrate | branch `merge-v1.0.0` = `a73c601e6` = merge of v1.0.0 (`b1b770d`, b12359) onto mxxm-t `cdc81dae4` + HIP top-k ballot fix + HIP `PAD_REFLECT_1D` launch-size fix; PR [mxxm-t/mx-llama.cpp#24](https://github.com/mxxm-t/mx-llama.cpp/pull/24) (supersedes #23, closed) |
| The build | tag `gfx906/v1.0.0/r0/master` = `df2c6e5c5` = substrate + the 18 Exabit commits; branches `master` = `gfx906-single` = `gfx906-multi` |
| Unbinned pool | tag `gfx906/v1.0.0/r0/candidates` = `5c01f4ffd`; branch `gfx906-candidates` |
| ROCm | 10.0.0-gfx906+20261018172427 (HIP 7.15.26333, rocBLAS 5.6) |
| Installed llama.cpp | deb `mx-llama-cpp-gfx906` 1.0.0+git20261114.df2c6e5c5.gfx906.20261018172427-1 from the ROLLING channel `mx-llama-cpp-gfx906` on `exabit-io/resize-amdgpu-bars` (per-version release `mx-llama-cpp-1.0.0-gfx906-rocm10.0` for the notes) |
| Installed PyTorch | deb `python3-torch-gfx906` 2.15.0+gfx906.20261018172427-1 (torchvision 0.30.0, torchaudio 2.11.0) from the ROLLING channel `python3-torch-gfx906` on `exabit-io/resize-amdgpu-bars`, with `python3-typing-extensions` 4.16.0-0exabit1 from the same channel (per-version release `pytorch-2.15.0-gfx906-rocm10.0`); work dir `/root/pytorch-gfx906-rocm10-2.15.0/` |
| RCCL topology | generated at boot by package `gfx906-rccl-topology`: its unit writes `/var/cache/gfx906-rccl-topology/topology.xml`, its conffile `/etc/rccl.conf` points RCCL at it (source `tools/gfx906-rccl-topology/`). Scripts do not set `NCCL_TOPO_FILE` |

Previous cycles: v0.7.0 `reports/2026-10-22-v070-rebase.md`, v0.9.0 `reports/2026-11-06-v090-rebase.md`; this cycle: `reports/2026-11-14-v100-rebase.md`.

## 1. Identities and access

- Commits: `Joshua <joshua@exabit.io>` (set as repo-local `user.name`/`user.email` in `/root/exabit-llama.cpp`). Never the personal name; check `git log --format='%an <%ae>'` on every rewritten range (the v0.5.0 candidates commits carried it).
- Push: `GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519_exabit -o IdentitiesOnly=yes"`. `gh` is logged in as the lead's account; it can act on `exabit-io`.
- Signing: Exabit, Inc. apt key `6043AD7B3533F615C15F48D0472316650F4BE230`, `GNUPGHOME=/root/exabit-apt-key/gnupg`.
- Commit trailer for AI-assisted commits: `Co-Authored-By: Claude ...`; PR bodies end with the Claude Code line. AGENTS.md's no-push/no-PR rules are scoped to ggml-org/llama.cpp and exempt forks; do not open anything against ggml-org.

## 2. Reconcile with mxxm-t first

mxxm-t merges upstream tags into his own master (`Merge tag 'bNNNNN' into mx-llama.cpp`). If he did since our last
cycle, our merge PR is obsolete and our commits sit on a dead base.

```sh
git fetch mxxm; git fetch exabit-mx; git fetch upstream --tags      # mxxm-t's own bNNNNN tags clash with upstream's: ignore the "would clobber" lines
gh pr list -R mxxm-t/mx-llama.cpp --state all                       # what happened to our last PR
git log --first-parent --oneline <old mxxm base>..mxxm/master       # what mxxm-t merged
git worktree add -b reconcile/mxxm-master /root/wt-reconcile mxxm/master
for c in $(git rev-list --reverse <old substrate>..exabit-mx/master); do git cherry-pick $c || break; done
```

Resolve each conflict by porting the patch's intent onto the new code, never by taking one side. Record each port in
the commit message (`git filter-branch --msg-filter` over the range, see `tools/release/portnotes.py`). Then
`git range-diff <old base>..exabit-mx/master mxxm/master..HEAD`: every `!` must be explained.

## 3. Merge the release into mxxm-t's master (the PR branch)

```sh
git worktree add -b merge-vX.Y.Z /root/wt-merge-vXYZ mxxm/master
git -c rerere.enabled=false merge --no-ff --no-commit vX.Y.Z         # rerere would replay stale resolutions from older cycles
```

Commit message format: `Merge llama.cpp vX.Y.Z (<sha>, bNNNNN) into master`, then the textual conflicts (file: how combined),
then "Merged without a textual conflict but changed here". Keep it ASCII. The PR is the pure merge, nothing else (mxxm-t did
not take #17's bundled RCCL default).

**Silent breaks: a clean merge or cherry-pick is not a correct one.** Before building, list the files both sides changed:

```sh
comm -12 <(git diff --name-only bOLD vNEW | sort) <(git diff --name-only bOLD mxxm/master | sort)
```

and read upstream's diff of every fork-heavy file in that list. The v0.7.0 cycle found four, all compiling cleanly:

| Class | v0.7.0 instance | Fix |
|---|---|---|
| Kernel layout changed under a fused patch | upstream split each GDN warp into 4 column segments; term 04's folded L2 norm still reduced over the whole warp (4x the sum) | reduce over `lanes_per_col`; re-derive any bit-exact claim (term 10) for the new layout and for the DPP reduction order |
| New upstream fast path that bypasses fork state | MMA chunked GDN kernel (#29353) writes only the final state and reads graph tensors | skip it when the snapshot ring keeps K > 1, and when the producer fold is active |
| New upstream feature on a path the fork replaced | LiliCorr (#30506) captures embeddings only in the eager DFlash inject; the fork defers injects | LiliCorr stays on the eager path |
| Fork shortcut that never covered a path | term 04's fold ignored by the chunked GDN kernel (KDA, 32-64 tokens) | folded calls stay on the recurrent kernel |

The v0.9.0 cycle added these classes (all compiled cleanly):

| Class | v0.9.0 instance | Fix |
|---|---|---|
| Upstream rewrote a kernel a fold lives in | float4 `rms_norm` variants (#29720): the fold-off path takes the vector kernel, the fold only existed in the scalar one, so folds-on stopped being bit-exact | implement the fold in both paths; run every shuffle on all lanes and gate only the stores |
| A shared numeric convention changed | Q8_1 sum is `d * sum(q)` (#31247); term 02's folded producer used the float sum | follow the producer the cache entry stands in for (`quantize_q8_1`) |
| Trailing struct member added | `mmq_args.fusion`; the fork's chunked launches built it without, which compiles and drops a fusion | grep every aggregate initializer of a struct upstream extended |
| Same name, two meanings | fork `layer_inp_dev` taps vs upstream's device buffer | rename one side |
| Same arch name, two formats | DeepSeek-V4.1 (`deepseek41`), fork vs upstream GGUF | take upstream's format (the converter writes it); port the fork's routing |
| Base moved mid-cycle | mxxm-t pushed `cdc81dae4` (column-thread GDN) while the PR was open | merge his master into the PR branch (force-push is blocked); re-check fork terms against his new code: term 04's folded calls had to skip the new kernel |
| Upstream bug on wave64 | grouped top-k fusion (#29181): HIP `__ballot_sync` ANDs with its mask | separate commit on the PR branch |

The v1.0.0 cycle added these (the first two compiled cleanly, the others were found by reading or by the tests):

| Class | v1.0.0 instance | Fix |
|---|---|---|
| Upstream feature that skips a path a fork feature feeds from | MTP device rows (#31927) skip the host extraction that fills the fork's deferred-prefill accum buffer: the replay would have injected zeros | sequence the two (deferred prefill first, device rows when its buffer is released) and refuse the combination at the API |
| Fork function that says it mirrors an upstream block | the KV-only MTP store mirrors `build_attn`'s store block, which gained the FP8 cache scales (#28392); its rope call used the removed `sections` array (#31683) | grep the fork for "mirrors" / copied blocks and diff upstream's version of each |
| Upstream changed what a shared helper outputs | `mm_ids_helper` writes tile specs, no expert bounds (#31047); the repacked MoE kernels read the bounds | keep an optional output; port the fork's chunked MMQ launches to the tile count and per-chunk tile specs |
| Both sides fixed the same bug | radix top-k determinism (#31084 vs 1b7d76ffc); unread scheduler inputs (#31419 vs 4d03828a1) | keep the stronger guarantee (fixed order; view-aware rule), drop the duplicate bookkeeping, say so in the merge message |
| Upstream moved a block to another file | graph structs to `graph.cuh` (#32001), NextN API to `llama.h` (#30565) | the conflict shows as a deleted block: move the fork's fields along, re-add includes that upstream dropped from files that still use fork-only API |
| New upstream test meets an old backend limit | `PAD_REFLECT_1D` with 2^24 rows: HIP rejects 2^32 threads along one grid dimension | separate commit on the PR branch; measure the limit with a minimal kernel before describing it |
| A test abort that looks like progress | `test-backend-ops` aborted at case 13,418 on all four dies; a count of OK lines kept looking like a run in progress | check the exit code and the `N/N tests passed` line, not the log size |

Stack the merge on the previous PR tip when that PR was not merged: `git merge vNEW` in a detached worktree at the old PR head gives
only the new conflicts (15 files for v1.0.0 instead of 27 + 15). Then build the real merge commit from that tree with
`git commit-tree TREE -p mxxm/master -p vNEW^{commit} -F merge-msg.txt` (revert the separate fix commits in the index first and
cherry-pick them back on top). `tools/release/fork-delta.py OLD_UPSTREAM OLD_FORK NEW_UPSTREAM` lists the files whose fork delta
changed in the merge (the ones edited) and the ones where it did not (read upstream's diff of the fork-heavy ones anyway).

`test-llama-context` and the other fixture tests: `tools/release/ctx-check.sh BUILD OUT` (direct runs, one worker). With all four dies
visible, `gemma4-dense` and `mistral-large4-moe` fail the sequence-copy cases because their KV-sharing layers land on different dies and
mxxm-t's KV replicas (de27e7509) are then in use; `LLAMA_KV_REPLICA=0` or `HIP_VISIBLE_DEVICES=0` passes all 144 models. This is in every
merge since v0.7.0 (reported in #24), so compare against that, not against zero failures. The dummy models get new random weights on
every run, so the count moves: gemma4-dense failed in 3 of 3 four-die runs, mistral-large4-moe in 2 of 3 (142 or 143 of 144).

Run `ctest -R test-llama-context` too: upstream's rollback cases assume the last ubatch's snapshots only, which the fork's ring
exceeds (fixed in the merge); the realloc debug check and the four-way parallel run fail on #22's tree as well (fork-side, open).

Also check the refactors that change an API the fork extends (v0.7.0: `llama_context::sched_new()`, the loader's per-file buffer
map removed in #30405): fold the fork's additions into the new structure.

## 4. Rebase the Exabit commits onto the merge

```sh
git worktree add -b master-vXYZ /root/wt-master-vXYZ merge-vX.Y.Z
for c in $(git rev-list --reverse mxxm/master..reconcile/mxxm-master); do git cherry-pick $c || break; done
```

Then re-check every Exabit commit against the new upstream paths (v0.7.0: term 04 had to skip the new MMA kernel).
`gfx906-single`/`-multi` = master until a profile winner exists. `gfx906-candidates`: rebuild it on `merge-vX.Y.Z` + the
RCCL commit, taking each term that is patch-identical to a master term (`git patch-id --stable`) from master's ported
commit, and the others from the old branch. Restore the `term NN-...` subjects and normalise authors afterwards.

## 5. Build and test

`tools/release/build.sh SRC BUILD LOG` (the tree's own defaults; ~2-4 min with ccache). Then, in this order, each with the
clamp watchdog and `gpu-test-env.sh` (macpro-fand pins the GPUs at 125 W). Check first that RCCL initialises
(no `NCCL init failed` in a `-sm tensor` log of a master build): a stale topology made every RCCL build fall back silently in Oct-Nov 2026:

| Test | Script | v0.7.0 values (reference) |
|---|---|---|
| suite on the merge (tree defaults, RCCL off) | `pr-test.sh BUILD OUT` | test-backend-ops 18644/18644 x4; PPL 16K/6 5.6122; `-sm layer -lm dio` 0 fallbacks; MTP 153/203, 69.2 t/s; Flash-Next generates |
| suite on master | `SKIP_TBO=1 pr-test.sh BUILD OUT` + test-backend-ops on two dies | 18644/18644 x2; **PPL 5.6264, bit-identical to the old master**; MTP 156/198, 74.4 t/s |
| decode-path folds | `fold-check.sh BUILD OUT` (ubatch **16**, `-v`); `fold-isolate.sh`; `kld-check.sh` | on 5.0677 / off 5.0568 (2K x 8); only the GDN L2 fold is not bit-exact: mean KLD 3.9e-4, same top 98.97% |
| direct I/O | `dio-check.sh` + `strace -e trace=openat` | O_DIRECT open, 0 fallbacks on every build |
| tolerance-edge flakes | re-run the single case 50x with `-o '<full case string>'` (inputs are random per run) | q5_1 n=1 k=256 at 5.13e-4 vs 5e-4: 0/150 on re-runs |

v1.0.0 reference values (same scripts): merge build test-backend-ops 20388/20388 x4, PPL 16K/6 5.6126, MTP 154/200 at 70.7-71.7 tok/s
steady state; master 20388/20388 x2, PPL 5.6093, folds on 5.0822 / off 5.0575 (all four unchanged from v0.9.0). `mtp-modes.sh BUILD OUT [NP]`
compares the MTP row paths (`LLAMA_MTP_DEV`, `LLAMA_MTP_DEFERRED_PREFILL`); a server log shows llama's own INFO lines (reserves, buffer sizes)
only with `-v`.

Gotchas: the fold check needs ubatch <= 16 (norm+q8 fold <= 16 rows; add-norm and GDN folds <= 64) and `-v` (ggml INFO lines
are hidden by default); `llama-perplexity --kl-divergence` prints its statistics on **stdout**; environment assignments in
front of a bash function call do not reliably reach the commands inside it (pass them through `env`); 16K PPL is a smoke
value (R2.2's 64K floor), comparable only build to build.

## 6. Push and PR

```sh
git push exabit-mx merge-vX.Y.Z:refs/heads/merge-vX.Y.Z
# PR through REST with head_repo: exabit-io owns two repos in this fork network, so gh pr create picks the wrong one
gh api --method POST repos/mxxm-t/mx-llama.cpp/pulls --input pr-req.json   # {title, head: "exabit-io:merge-vX.Y.Z", head_repo: "exabit-io/mx-llama.cpp", base: "master", body}
git tag -a gfx906/vX.Y.Z/r0/{merge,master,candidates} ...                  # message: what it is + the verification numbers
git push exabit-mx --force-with-lease=refs/heads/master:<old sha> ... master-vXYZ:refs/heads/master ...  # every branch with its own lease
```

Then the BRANCHES.md commit on `gfx906-candidates` (it names the PR number) and its push. Pushing `master` triggers
upstream's workflows on the fork (that is where the `bNNNNN` tags on exabit-io come from).

## 7. Packages

| Step | mx-llama.cpp | PyTorch |
|---|---|---|
| build | `build-mx-llama-deb.sh <clean worktree of master>` (refuses a dirty tree; `-march=native`, DT_RPATH check, Depends from ELF NEEDED + the matching `-gfx906` kernel packages) | `build-torch.sh` (clone `vX.Y.Z` with submodules into `src/` first) then `build-deb.sh` |
| test before publishing | `apt-get install ./deb`, then `SKIP_TBO=1 pr-test.sh /opt/mx-llama.cpp OUT`; rerun MTP on an idle host (a parallel compile cut it from 74.4 to 54.7 t/s) | `stack_check.py` with the build venv, then with `/usr/bin/python3 -I` after `apt-get install ./deb`; compare with the previous package (same check) |
| publish | `publish.sh LLAMA_DEB TORCH_DEB` (signs the debs, tags resize-amdgpu-bars `origin/main`, runs `/root/local-apt/github-release-pkg.sh` once per package) | same call |
| dogfood | point `/etc/apt/sources.list.d/*.sources` at the new releases, `apt-get clean; apt-get install --reinstall ...`, verify InRelease signature -> Packages SHA256 -> deb SHA256 -> embedded origin signature, rerun the checks | same |

Since v1.0.0 `mx-llama-cpp-gfx906` has a rolling channel (release `mx-llama-cpp-gfx906` on `exabit-io/resize-amdgpu-bars`, channel dir
`/root/local-apt/channel-mx-llama-cpp-gfx906`, notes `/root/local-apt/mx-llama-cpp-gfx906-channel-notes.md`): `publish-100.sh DEB` signs,
tags, publishes the per-version release and then adds the deb to the channel. Machines keep the channel line and never a per-version one.
Before publishing, `deb-test.sh OUT` takes a reference on the previous package, installs the new deb from the file and runs the suite on
`/opt/mx-llama.cpp`. After publishing, `dogfood-100.sh OUT` writes the channel source line, reinstalls from the channel and verifies
InRelease signature -> Packages SHA256 -> deb SHA256 -> embedded origin signature. Build the deb from a clean detached worktree of the final `master` (`git worktree add --detach /root/wt-deb-vXYZ master-vXYZ`).

Since 2.15.0 `python3-torch-gfx906` has a rolling channel too (release `python3-torch-gfx906`, channel dir
`/root/local-apt/channel-python3-torch-gfx906`, notes `/root/local-apt/python3-torch-gfx906-channel-notes.md`). The cycle, all in
`/root/pytorch-gfx906-rocm10-X.Y.Z/` (copies in `tools/release/`): clone `vX.Y.Z` with submodules into `src/pytorch`,
`build-torch-2.15.0.sh`, `build-torch-deb-2.15.0.sh`, `torch-deb-test.sh OUT` (reference on the installed package, install from the
files, dependency checks, `stack_check.py` and `torch-ops-check.py`), `publish-torch.sh DEB` (per-version release, then the channel),
`dogfood-torch.sh OUT`. Read the upstream release notes first (`gh api repos/pytorch/pytorch/releases/tags/vX.Y.Z -q .body`): the
ROCm, Distributed, Release Engineering and Developers sections name what changes for a source build.

PyTorch traps found on 2.15.0:
- **Compare the wheel's `Requires-Dist` with what noble ships before building the deb, and import the staged tree with the system
  interpreter** (`sys.path.insert(0, STAGE/usr/lib/python3/dist-packages)`). 2.15 needs typing_extensions >= 4.13.0 at `import torch`
  (noble: 4.10.0); the build venv has its own copy and passes everything. The channel carries a `python3-typing-extensions` backport
  (`build-typing-extensions-deb.sh`: the upstream wheel, Ubuntu's package name and layout) and the deb depends on it with the bound.
  When PyTorch raises the bound again, rebuild the backport from the version in the new build venv.
- A package that depends on a second package of ours: put both debs in the per-version release and in the channel (`ARCHES="amd64 all"`),
  install both files in one `apt-get install` for the test, and check that apt refuses the main deb alone.
- `amd_smi` is a required CMake package for ROCm 7.14 and later; ROCm 10.0 has it. `libtorch_hip.so` links `libamd_smi.so`.
- 2.15's extra distributed backends (`nccl-lazy`, `nccl-legacy`, `nccl2`) are registered through the wheel's entry points
  (`torch.distributed.backends`), not listed in `Backend.backend_list`: the deb must ship `torch-*.dist-info/entry_points.txt` (it does),
  and a test must look there to know whether a backend exists.
- A check that compares GPU and CPU index results needs tie-free input (rows that are permutations): random floats repeat often
  enough in 5,000 draws for an unstable sort to differ legitimately.
- The published asset is the reference for an old deb, not the local file: the local 2.13.0 deb differed from its release asset.
  `publish-torch.sh` compares the channel copies with the digests GitHub reports.

PyTorch traps found on 2.14.1:
- 2.14 builds through scikit-build-core: `python -m build --wheel --no-isolation`; `setup.py bdist_wheel` is gone.
- With `-march=native`, PyTorch's FindAVX accepts its empty first flag and caches `CXX_AVX512_FLAGS=' '`; only fbgemm reads it, and its AVX512 sources lose `-mavx512bf16`. Seed `CMAKE_ARGS` with the value a non-native build computes (`-mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512bf16 -mfma`); fbgemm gates the BF16 code on cpuinfo at run time.
- `import torchvision` needs Pillow: the deb depends on `python3-pil`.
- noble ships sympy 1.12 and setuptools 68; torch declares >= 1.13.3 / >= 77. Irrelevant without Triton (`torch.compile` is unavailable anyway), but pip warns.
- Verify `-march=native` reached the compiler: `grep -c -- -march=native build/compile_commands.json`.

Packaging trap: in bash, `${VAR//~/.}` tilde-expands the pattern to `$HOME`.

## 8. Memory and documents to update at the end

`CLAUDE.md` (code paragraph), `README.md` (Status), `REQUIREMENTS.md` change log (factual pointer only), `RE-RE-SURVEY-ACTION-PLAN.md`
section 2 (both copies: here and `/root/`), `reports/README.md`, the agent memory (`mx-llama-cpp-v070-pr`, `exabit-package-publishing`,
`macpro-rocm-gfx906-install`).
