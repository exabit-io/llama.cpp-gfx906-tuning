# Release runbook: taking a llama.cpp release through the gfx906 stack

Written 2026-10-22, after the v0.7.0 cycle (reconcile with mxxm-t, merge, rebase, PR, packages). Follow it in order for the
next upstream release. Every script named here is in `tools/release/` (box-side copies; the working copies and their logs
from the v0.7.0 cycle are in `/root/rebase-v070/` on macpro2019-01). Report: `reports/2026-10-22-v070-rebase.md`.

## 0. State after the v1.0.0 cycle (2026-11-14)

| What | Where / value |
|---|---|
| Code repository | `github.com/exabit-io/mx-llama.cpp`; local clone `/root/exabit-llama.cpp`, remote `exabit-mx`; upstream fork remote `mxxm` (mxxm-t/mx-llama.cpp), `upstream` (ggml-org) |
| Pure substrate | branch `merge-v1.0.0` = `a73c601e6` = merge of v1.0.0 (`b1b770d`, b12359) onto mxxm-t `cdc81dae4` + HIP top-k ballot fix + HIP `PAD_REFLECT_1D` launch-size fix; PR [mxxm-t/mx-llama.cpp#24](https://github.com/mxxm-t/mx-llama.cpp/pull/24) (supersedes #23, closed) |
| The build | tag `gfx906/v1.0.0/r2/master` = `224b73e24` = substrate + the 18 Exabit commits (`gfx906/v1.0.0/r0/master` = `df2c6e5c5`) + the LLVM 24 rule for the head-256 tile instance (`r1` = `290fc642a`, threshold 90) + its threshold moved to 85 (MTP decode at depth); branches `master` = `gfx906-single` = `gfx906-multi` |
| Unbinned pool | tag `gfx906/v1.0.0/r0/candidates` = `5c01f4ffd`; branch `gfx906-candidates` |
| ROCm | runtime **10.1.0-gfx906+20261018172427** (HIP 7.16.26385, LLVM 24, RCCL 2.30.7) in `/opt/rocm/core-10.1`; 10.0.0-gfx906+20261018172427 (HIP 7.15.26333, LLVM 23) is still installed in `/opt/rocm/core-10.0` (reference builds; the llama package is compiled by 10.1 since 2026-11-15) |
| Installed llama.cpp | deb `mx-llama-cpp-gfx906` 1.0.0+git20261116.224b73e24.rocm10.1.gfx906.20261018172427-1 (compiled by and run on ROCm 10.1, no global `-march=native`) from the ROLLING channel `mx-llama-cpp-gfx906` on `exabit-io/resize-amdgpu-bars` (per-version release `mx-llama-cpp-1.0.0-gfx906-rocm10.1-r2`; `...-rocm10.1-r1` is `290fc642a`, the rule at 90; `...-rocm10.0-r1` the same commit by ROCm 10.0's compiler; `...-rocm10.1` is the first 10.1 build, `df2c6e5c5` compiled by ROCm 10.0's toolchain; `...-rocm10.0` the 10.0 build) |
| Installed PyTorch | deb `python3-torch-gfx906` 2.15.0+rocm10.1.gfx906.20261018172427-1 (torchvision 0.30.0, torchaudio 2.11.0) from the ROLLING channel `python3-torch-gfx906`, with `python3-typing-extensions` 4.16.0-0exabit1 and `python3-sympy` 1.14.0-2~24.04.exabit1 from the same channel (per-version release `pytorch-2.15.0-gfx906-rocm10.1`); work dir `/root/pytorch-gfx906-rocm10.1-2.15.0/` |
| RCCL topology | generated at boot by package `gfx906-rccl-topology`: its unit writes `/var/cache/gfx906-rccl-topology/topology.xml`, its conffile `/etc/rccl.conf` points RCCL at it (source `tools/gfx906-rccl-topology/`). Scripts do not set `NCCL_TOPO_FILE` |

Previous cycles: v0.7.0 `reports/2026-10-22-v070-rebase.md`, v0.9.0 `reports/2026-11-06-v090-rebase.md`; this cycle: `reports/2026-11-14-v100-rebase.md`.

**v1.1.0 cycle (2026-11-26): in progress. The merge is pushed and offered as PR mxxm-t#25 (#24 closed); master, the candidates, the tags and the package are not out, so the table above is still the shipped state.** Report `reports/2026-11-26-v110-rebase.md`; record, scripts and the open list: `tools/release/v110/README.md` (`/root/rebase-v110/` on the box).

**Direction since 2026-11-26: this fork is its own project.** None of the merges offered to mxxm-t was taken (PRs #17, #22, #23, #24; #25 open). The fork follows upstream llama.cpp releases itself and pulls in what serves its goals: mxxm-t's gfx906 patches stay, and his later changes keep being read and taken on their merit. What ends: waiting on his tree, resolving in his favour only to stay mergeable, and opening merge pull requests (none unless the lead asks). Sections 2 and 3 below were written for the bound procedure: read section 2 as "compare with his tree and take what is good" and skip the pull request of section 3 and 7. The condition (no major, statistically significant performance penalty against his fork) and its measurement: `reports/2026-11-26-v110-rebase.md` section 10.

**Learned on 2026-11-26/27 (v1.1.0 cycle), for every later cycle:** (1) "replay and snapshots write the same text" is a property of the MERGE; on master the two differ by rounding for two independent reasons (the GDN producer fold reaches snapshot operations only; the DPP lane sums make the GDN kernel's live and replay loops round differently): check the rollback by its fixture cases, not by text equality across the two modes. (2) A single `SOLVE_TRI` miss near 1e-7 on one die is the test's tolerance against random 128 x 128 systems (0 in 96 repeats). (3) `deepseek41` has two GGUF layouts and two implementations in master; upstream's fixture case `comp_padding` is the check that the upstream-layout path reuses its graph. (4) A working RCCL logs nothing: "linked, forced by the environment, none of its three failure warnings" is the evidence (`tools/release/v110/sub-exchange.sh`). (5) When two arms that should agree do not, trace it with controls whose reading is written before the run: fold switches, the other build option, a bisection, a one-line opt-out (`reports/2026-11-26-v110-rebase.md` section 11).

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

**When he merged upstream himself and our last PR was not taken (v1.1.0, 2026-11-26):** compare the two handlings
before anything else. Merge the upstream point HE merged into OUR last merge in a scratch worktree
(`git worktree add --detach /root/wt-cmp <our PR tip>; git merge --no-commit <his upstream parent>`), then
`git diff <his merge commit> -- <file>` in that worktree shows his handling against ours with upstream's own changes
cancelled; the files that conflict there are compared as fork deltas. Give the reading to several readers with one
brief (`tools/release/v110/README.md`, `data/raw/2026-11-26/v110/cmp/BRIEF.md`: verdict labels, "read before you
claim"), and RUN his master through the suite: the 68 failing `TOPK_MOE` cases were in the test output before any
reader named their cause. The new PR starts from his tree; where the handlings differ it has his, and what ours
fixed that his lacks goes on the branch as separate commits, each with a run that shows it.

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

The v1.1.0 cycle added these:

| Class | v1.1.0 instance | Fix |
|---|---|---|
| Upstream replaces a mechanism the fork has its own answer to | #31669: replay instead of snapshot rollback for the delta-net models, snapshot branch removed; the fork has the ring and plane-indexed snapshots | keep both, one per context, behind ONE predicate both sides call (`ring_supported()`); the fork's kernels take only ops without the new inputs; decide the default by text equality with the kernel held fixed (`rs-modes2.sh`), memory and speed |
| A hunk that applies cleanly inside a look-alike block | term 01's rows-per-block cases landed in upstream's new ORIN branch of `calc_rows_per_block`, whose tail reads like the GCN one; gfx906 silently fell back to upstream's values | after a port, compare every changed FUNCTION with the old commit's version, not only the conflicts; a second reader found this one |
| Upstream removes a test or a branch the fork still uses | the K > 1 gated delta net cases of `test-backend-ops`, deleted with upstream's snapshot branch | restore them as fork tests; check `git diff <his> -- tests/` for deletions |
| Upstream renames a file a per-file build rule names | `fattn-tile-instance-dkq256-dv256.cu` -> `...-f16-f16-dkq256-dv256.cu` (#32901); CMake ignores a property on a source that is not in the target | the rule globs and fails the configure on a miss; grep the build for the flag after every cycle |
| A derived flag set before a fork clamp | upstream sets `cparams.rs_replay` before the fork clamps `n_rs_seq` for a small `n_ubatch` | compute derived flags after the last clamp |
| Upstream turns member code into free functions | `build_attn_iswa_impl(gctx, ...)`: the fork's member helpers no longer in scope (compile error), or silently bypassed where upstream's direct call still compiles | grep the refactored file for direct calls the fork routes through a helper |
| A fork kernel written for 32-lane rows after upstream widened the row | fused top-k softmax: default-width reductions on 64-lane rows (#31617) | every reduction names the row width; run `test-backend-ops` on HIS tree too |
| A new upstream test model meets a kernel instance that does not exist | `hy_v4_flash`, 576/512 with GQA ratio 2: abort in the tile launcher, stock aborts too | the selection returns NONE; an abort in a fixture run hides every model after it |
| The numbers move upstream | PPL 5.6186 -> 5.6070 on stock between b12500 and v1.1.0 | build stock at both releases and attribute (`numerics.sh`) before reading a changed PPL as a merge fault |

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
The pool has neither the `max-ilp` default nor the LLVM 24 tile rule: build it with ROCm 10.0's compiler
(`tools/rocm101/build-rocm.sh SRC BUILD LOG /opt/rocm/core-10.0`). Its head-256 tile object under clang 24 was read
statically on 2026-11-17 (`tools/rocm101/cand-tilescan.sh`): clean only with `max-ilp` AND threshold 85 together;
that setting has not been timed (`reports/2026-11-14-rocm-10.1.md` section 7).

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

Since v1.1.0 (`tools/release/v110/`): `queue-final.sh BUILD TAG SRC` is the whole verification of a tree (suite, fixtures
with the default rollback and with `LLAMA_RS_REPLAY=1`, MTP row paths, `mtp-carry.sh`, `rs-modes3.sh`; `FOLD=1` adds the
fold check). **No build and no other host load while the dies are loaded, also during correctness runs, and read the
watchdog's clock column (`card0:NNNNMHz/100%`) after every GPU job:** on 2026-11-26 the dies latched at 1000 MHz
within seconds of a build starting beside a GPU test, and two hours of speeds were void before a watchdog said so.
Do not rewrite a source file while a build compiles it either (clang dies with a Bus error).

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
| build | `build-mx-llama-deb.sh <clean worktree of master>` (refuses a dirty tree; CMake Release defaults for the host code with `GGML_NATIVE=ON`, no global `-march=native` since 2026-11-15; DT_RPATH check, Depends from ELF NEEDED + the matching `-gfx906` kernel packages) | `build-torch.sh` (clone `vX.Y.Z` with submodules into `src/` first) then `build-deb.sh` |
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

### A new ROCm stream (10.1, 2026-11-14)

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
   read the kernel's register use from the code object. Spill counts alone did NOT predict speed across scheduler
   strategies in the 10.1 case. What settled it: `codegen/ircross.py` (each compiler's optimized bitcode through the
   other's code generator: middle end or code generator?), `drv-test.sh` (the driver's offload pipeline, with and
   without LTO), then single code-generator options on the one object, and `gpu-opt-confirm.sh` (reference, plain new
   compiler, candidate; n = 3 shuffled at 4 x 64K and 1 x 128K; rule in its header) before any package is built.
   When the code generator is the cause, find the commit with `llc` alone (`tools/rocm101/llvm-bisect/`, 2026-11-16):
   `clang --version` of a ROCm compiler names its ROCm/llvm-project commit; `prep.sh` makes a blobless clone and the merge
   base, `step.sh` builds `llc` (AMDGPU only, own ccache directory, 4-10 min) at the checked-out commit, compiles the
   kernel's optimized bitcode (from `codegen/ircross.py`) and applies a static criterion (`codegen/loopstat.py`: reloads
   in the hot loop), `run.sh` validates both endpoints and runs `git bisect` (13 steps for 10,434 commits, 1.2 h).
   **Before writing any report, check whether upstream already fixed it** (`confirm.sh`: the one-line revert on the ROCm
   commit, and `llc` at llvm/llvm-project `main`; `git log <ROCm's upstream base>..main -- <the files>` finds the fix
   faster than a second bisect; `git branch -r --contains <fix>` tells which ROCm release branches have it). Then time
   the objects: `codegen/llccross.py` puts an `llc` object into a library through ROCm 10.0's own driver jobs
   (`bisect-screen.sh`, `fixrule-screen.sh`). The 10.1 case: a bug from 2026-08-03, fixed upstream on 2026-08-19, with
   ROCm 10.1's branch cut on 2026-08-13 (report `2026-11-14-rocm-10.1.md` sections 4.2, 4.3).
5. The packages: version `...rocm<stream>.gfx906.<stamp>` (two streams can carry one stamp); the llama deb takes
   `ROCMSTREAM` (runtime: Depends and `DT_RPATH`) and `COMPILERSTREAM` (toolchain, default the runtime's; a source
   without the LLVM 24 tile rule is refused with clang 24 or newer) separately; test the package
   (`deb-test-llama-101.sh`), then accept it on the installed package with the same rule (`rocm-ab2.sh`; `rocm-ab3.sh`
   when the package's commit differs from the reference's only in the HIP CMake file).
   A difference between the package and a source build is located by swapping shared libraries, not by rebuilding:
   `pkg-swap.sh` preloads all of the package's libraries by full path with one group taken from the other build (6 min
   per arm at 4 x 64K; ranges of 3-4 runs separated 0.4 %). That is how the deb's host flag was found.
   `flag-screen.sh` screens host compiler flags the same way (variants built by `flag-builds.sh`). Its result of
   2026-11-15: nothing beats CMake's Release defaults with `-march=native` only where `GGML_NATIVE` puts it; a global
   `-march=native` costs 0.4 % of 4 x 64K decode, 512-bit vectorization 0.6-0.8 %, `-O2` 1.3 %; LTO and PGO gain nothing.
   `build-mx-llama-deb.sh` no longer sets `-DCMAKE_C_FLAGS`/`-DCMAKE_CXX_FLAGS=-march=native` (2026-11-15); the next package is
   accepted with `rocm-ab2.sh`. `rt-swap.sh` does the same for the ROCm libraries (which library of a runtime stream costs
   decode): every ROCm library preloaded by full path, one group taken from the other stream. One library preloaded alone is
   not a swap: the ROCm libraries carry `DT_RPATH=$ORIGIN` and pull their whole tree in.
6. Before publishing a package that pulls the new runtime onto nodes, test the old packages on it (`suite-mixed.sh`):
   that is the state of every node between the two upgrades.
7. A build of the same commit for the OLDER stream (`ROCMSTREAM=10.0 build-mx-llama-deb.sh SRC`, 2026-11-16), for nodes
   that stay on it: its version sorts below the newer stream's build of that commit, and it must not become the
   channel's newest version (`publish-llama-100r1.sh` refuses that). Test it (`deb-test-llama-100.sh`: the package must
   load its own stream although `/opt/rocm` and `LD_LIBRARY_PATH` name the other), accept it against the reference on
   the same runtime (`rocm-ab4.sh`), publish, then `dogfood-llama-100.sh OUT VERSION`: install by version from the
   channel, verify the chain, and install the newest again so the box ends on what it runs. The channel notes carry
   the install-by-version and `apt-mark hold` lines. Comparing GPU code of clang 23 builds from two source directories:
   `devcmp.py` reports 136 objects as different because the error printer `no_device_code` embeds `__FILE__`; use
   `fncmp.py` and read which functions differ.
8. **A per-object compile option is read on every kernel of the object, and accepted on every kernel the service
   runs** (2026-11-16; report sections 5.10, 5.11). The tile rule of `-r1` (threshold 90) was chosen on the prefill
   kernel and accepted with `llama-batched-bench`, which runs two of the object's 34 kernels (`<16,2>` prefill, `<1,2>`
   decode). It put 20 reloads into the loop of `<4,2>`, the kernel that verifies three drafted tokens, and cost MTP
   decode 1.8 % at depth; `-r2` (threshold 85) is the correction. Before timing an option: `codegen/tilescan.py OBJ...`
   (every kernel of a tile object: reloads in loops / loop size; `codegen/trycc.py` makes the objects) and
   `codegen/loopscan.py BUILD_A BUILD_B` (every device function of two builds). Timing MTP at depth:
   `mtp-depth2.sh` (swap libraries under the installed `llama-server`, 117.7K-token chat prompt, 4.7 min per cell,
   decode compared only between cells with the same text and draft counts). A package whose GPU code changes in the
   tile object is accepted with BOTH `rocm-ab5.sh` (design points, 3 h) and `mtp-accept.sh` (the installed package
   against the previous package's GPU library, 45 min); the confirmation before the build is `tile-thr-confirm.sh`.
   `hostcc-screen.sh` and `mtp-accept.sh` compare against the package that is installed, so they run before the new
   deb is installed and after it, respectively. The package ships no tests: `tbo-r2.sh` runs the previous source
   build's `test-backend-ops` with the installed package's `libggml-hip.so` preloaded (one process per die, the
   mapped library read back; 20388/20388 x4 for `-r2`). The GPU code of the package and of a source build from
   another directory is compared with `codegen/maskcmp.py` (`fncmp.py` alone reports every function that references
   a string). `-r2`'s whole sequence, in order, is the row block of 2026-11-16 in `tools/rocm101/README.md`;
   `dogfood-llama-101.sh` takes an absolute output directory.

Backports of Ubuntu packages (`tools/backports/build-backport.sh DSC BUILD_DEPS`): fetch the newer series' source with
`apt-get source` through a private apt configuration (verified against the Ubuntu archive key), rebuild in the clean noble
chroot `/var/cache/exabit-backports/noble-buildd` (mmdebstrap), version `<newer>~24.04.exabit1`, publish into the channel
of the package that needs it. Check the newer package's binary relations first: Ubuntu 26.04's setuptools breaks
wheel < 0.46 and depends on eight de-vendored libraries, which is why it was not backported.

PyTorch traps found on 2.15.0:
- **A declared requirement that noble does not satisfy is tested, not assumed harmless** (`torch-pydeps-check.py`: export with dynamic
  shapes, symbolic tracing, `torch.compile` on the CPU, C++ extensions through ninja and through setuptools). sympy 1.12 broke the first
  three in 2.15; the first 2.15.0 notes said nothing needed the newer one and had to be corrected.
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

Build traps found on v1.1.0 (both handled in `build-mx-llama-deb.sh` and `tools/release/v110/build-rocm.sh`):
- A `gcc-N` without `libstdc++-N-dev` on the build box: the ROCm clang selects the newest GCC directory and the configure
  ends at `'cmath' file not found`. Pin with `-DCMAKE_HIP_FLAGS=--gcc-install-dir=$(dirname $(c++ -print-libgcc-file-name))`.
- The server's web UI is downloaded at build time by build number (`b<commit count>`). A fork's count names no upstream
  build; the fallback `latest` fails the asset check once it is newer than the tree. Set `HF_UI_VERSION` to the b tag on
  the merged release's commit (`git tag --points-at vX.Y.Z^{commit}`), and check the build log's `resolved:` line.

Packaging trap: in bash, `${VAR//~/.}` tilde-expands the pattern to `$HOME`.

## 8. Memory and documents to update at the end

`CLAUDE.md` (code paragraph), `README.md` (Status), `REQUIREMENTS.md` change log (factual pointer only), `RE-RE-SURVEY-ACTION-PLAN.md`
section 2 (both copies: here and `/root/`), `reports/README.md`, the agent memory (`mx-llama-cpp-v070-pr`, `exabit-package-publishing`,
`macpro-rocm-gfx906-install`).
