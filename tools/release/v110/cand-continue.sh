#!/bin/bash
# cand-continue.sh -- in /root/wt-candidates-v110: rebuild the unbinned pool (gfx906-candidates, 5c01f4ffd at v1.0.0) on
# merge-v1.1.0, in the pool's order: a pool commit that is patch-identical to a v1.0.0 master commit is taken from
# master-v110's port of it (same subject); the pool-only terms (17, 18, 19, 21, 22, 27) and the docs commits are
# cherry-picked from the old pool. Finishes a pick in progress first; stops at the first conflict.
set -u
cd /root/wt-candidates-v110 || exit 1
if [ -f "$(git rev-parse --git-dir)/CHERRY_PICK_HEAD" ]; then
  git diff --name-only --diff-filter=U | grep -q . && { echo "unresolved files:"; git diff --name-only --diff-filter=U; exit 1; }
  git -c core.editor=true cherry-pick --continue > /dev/null || exit 1
fi
declare -A M
while read -r h s; do M["$s"]=$h; done < <(git log --format='%h %s' merge-v1.1.0..master-v110)
done_subjects=$(git log --format=%s merge-v1.1.0..HEAD)
for c in $(git rev-list --reverse a73c601e6..5c01f4ffd); do
  s=$(git log -1 --format=%s $c); key="$s"
  [ "${s:0:8}" = "term 06-" ] && key="tp-allreduce: GGML_TP_AR_MAX_NE explicit size gate for (ported to b10912)"
  grep -qxF "$s" <<< "$done_subjects" && continue
  src=${M[$key]:-$c}; from=master-v110; [ "$src" = "$c" ] && from=pool
  if git -c rerere.enabled=false cherry-pick $src > /root/rebase-v110/cand-$(git rev-parse --short $c).out 2>&1; then
    [ "$from" = master-v110 ] && [ "$key" != "$s" ] && git commit -q --amend -m "$s" -m "$(git log -1 --format=%b)"
    echo "ok   ($from) ${s:0:80}"
  else
    echo "STOP ($from $(git rev-parse --short $src)) ${s:0:80}"
    for f in $(git diff --name-only --diff-filter=U); do echo "  $f: $(grep -c '^<<<<<<<' $f) hunks"; done
    exit 2
  fi
done
echo "ALL PICKED: $(git rev-list --count merge-v1.1.0..HEAD) commits on merge-v1.1.0"
