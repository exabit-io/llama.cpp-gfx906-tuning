#!/usr/bin/env python3
"""fork-delta.py OLD_UPSTREAM OLD_FORK NEW_UPSTREAM [NEW_FORK]

Run inside the merge worktree after resolving a release merge. For every file that both the fork and upstream changed,
compare the fork's own delta before the merge (OLD_UPSTREAM..OLD_FORK) with its delta after it (NEW_UPSTREAM..NEW_FORK,
default: the working tree). A file whose delta is unchanged kept every fork line; a file whose delta changed is one that
the merge edited, by hand or automatically, and must be read. Neither proves the code is right: an unchanged delta can
sit on upstream code that changed underneath it (the silent-break classes of RELEASE-RUNBOOK.md), so the upstream diff
of the fork-heavy files still has to be read.

Example (v1.0.0 cycle): fork-delta.py v0.9.0 merge-v0.9.0 v1.0.0
"""
import collections
import subprocess
import sys


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def delta(a, b, path):
    args = ["diff", "-U0", a] + ([b] if b else []) + ["--", path]
    out = git(*args)
    return [l for l in out.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---"))]


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    old_up, old_fork, new_up = sys.argv[1:4]
    new_fork = sys.argv[4] if len(sys.argv) > 4 else None
    up = set(git("diff", "--name-only", old_up, new_up).split("\n"))
    fork = set(git("diff", "--name-only", old_up, old_fork).split("\n"))
    both = sorted(f for f in up & fork if f)
    same, changed = [], []
    for f in both:
        old = collections.Counter(delta(old_up, old_fork, f))
        new = collections.Counter(delta(new_up, new_fork, f))
        if old == new:
            same.append(f)
        else:
            changed.append((sum((old - new).values()), sum((new - old).values()), f))
    print(f"{len(both)} files changed by both sides; fork delta unchanged in {len(same)}, changed in {len(changed)}:")
    for lost, gained, f in sorted(changed, key=lambda t: -(t[0] + t[1])):
        print(f"  {f}: {lost} fork lines gone, {gained} new")
    print("unchanged (read upstream's diff of the fork-heavy ones anyway):")
    for f in same:
        n_up = git("diff", "--numstat", old_up, new_up, "--", f).split()
        print(f"  {f}: upstream +{n_up[0]} -{n_up[1]}")


if __name__ == "__main__":
    main()
