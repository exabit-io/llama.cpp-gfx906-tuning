# Release runbook: taking a llama.cpp release through the gfx906 stack

Written 2026-10-22, after the v0.7.0 cycle (reconcile with mxxm-t, merge, rebase, PR, packages). Follow it in order for the
next upstream release. Every script named here is in `tools/release/` (box-side copies; the working copies and their logs
from the v0.7.0 cycle are in `/root/rebase-v070/` on macpro2019-01). Report: `reports/2026-10-22-v070-rebase.md`.

## 0. State after the v0.7.0 cycle

| What | Where / value |
|---|---|
| Code repository | `github.com/exabit-io/mx-llama.cpp`; local clone `/root/exabit-llama.cpp`, remote `exabit-mx`; upstream fork remote `mxxm` (mxxm-t/mx-llama.cpp), `upstream` (ggml-org) |
| Pure substrate | branch `merge-v0.7.0` = `342507407` = mxxm-t master `c4d12e2c0` (upstream b11553) + one merge of v0.7.0 (`620d222`, b11669); PR [mxxm-t/mx-llama.cpp#22](https://github.com/mxxm-t/mx-llama.cpp/pull/22) |
| The build | `master` = `gfx906-single` = `gfx906-multi` = `ef70ef51e` = substrate + 18 Exabit commits (RCCL default, AR size gate, FA_QUANTS=all, AR 20481, terms 02 03 04 07 08 10 12 23 28 01 15 05 09, max-ilp) |
| Unbinned pool | `gfx906-candidates` = `9a901c33f` (substrate + RCCL + 20 terms + BRANCHES.md) |
| Tags | `gfx906/v0.7.0/r0/{merge,master,candidates}`, `gfx906/b11553/r0/master` (same 18 commits on mxxm-t's b11553 master); older states `gfx906/v0.5.0/*` |
| ROCm | 10.0.0-gfx906+20261018172427 (HIP 7.15.26333, rocBLAS 5.6), apt repo `s3.arkprojects.space/apt-gfx906`, pinned `amdrocm*` to `10.0.0-gfx906*` |
| Installed llama.cpp | deb `mx-llama-cpp-gfx906` 0.7.0+git20261022.ef70ef51e.gfx906.20261018172427-1 in `/opt/mx-llama.cpp` (DT_RPATH, no `LD_LIBRARY_PATH` needed or possible) |
| Installed PyTorch | deb `python3-torch-gfx906` 2.14.1+gfx906.20261018172427-1 (torchvision 0.29.0, torchaudio 2.11.0), system `python3` |
| Package releases | `github.com/exabit-io/resize-amdgpu-bars/releases/tag/mx-llama-cpp-0.7.0-gfx906-rocm10.0` and `.../pytorch-2.14.1-gfx906-rocm10.0` (signed flat apt repos); box sources in `/etc/apt/sources.list.d/{mx-llama-cpp-gfx906,pytorch-gfx906}.sources` |

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
clamp watchdog and `gpu-test-env.sh` (macpro-fand pins the GPUs at 125 W):

| Test | Script | v0.7.0 values (reference) |
|---|---|---|
| suite on the merge (tree defaults, RCCL off) | `pr-test.sh BUILD OUT` | test-backend-ops 18644/18644 x4; PPL 16K/6 5.6122; `-sm layer -lm dio` 0 fallbacks; MTP 153/203, 69.2 t/s; Flash-Next generates |
| suite on master | `SKIP_TBO=1 pr-test.sh BUILD OUT` + test-backend-ops on two dies | 18644/18644 x2; **PPL 5.6264, bit-identical to the old master**; MTP 156/198, 74.4 t/s |
| decode-path folds | `fold-check.sh BUILD OUT` (ubatch **16**, `-v`); `fold-isolate.sh`; `kld-check.sh` | on 5.0677 / off 5.0568 (2K x 8); only the GDN L2 fold is not bit-exact: mean KLD 3.9e-4, same top 98.97% |
| direct I/O | `dio-check.sh` + `strace -e trace=openat` | O_DIRECT open, 0 fallbacks on every build |
| tolerance-edge flakes | re-run the single case 50x with `-o '<full case string>'` (inputs are random per run) | q5_1 n=1 k=256 at 5.13e-4 vs 5e-4: 0/150 on re-runs |

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
