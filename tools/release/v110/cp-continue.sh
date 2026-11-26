#!/bin/bash
# cp-continue.sh -- in /root/wt-master-v110: finish the cherry-pick in progress (everything staged), then go on through
# the Exabit commits a73c601e6..224b73e24 that are not yet picked, stopping at the first conflict.
set -u
cd /root/wt-master-v110 || exit 1
if [ -f "$(git rev-parse --git-dir)/CHERRY_PICK_HEAD" ]; then
  git diff --name-only --diff-filter=U | grep -q . && { echo "unresolved files:"; git diff --name-only --diff-filter=U; exit 1; }
  grep -lE '^(<<<<<<< |>>>>>>> )' $(git diff --cached --name-only) 2>/dev/null | grep -q . && { echo "conflict markers left"; exit 1; }
  git -c core.editor=true cherry-pick --continue > /dev/null || exit 1
fi
done_subjects=$(git log --format=%s merge-v1.1.0..HEAD)
for c in $(git rev-list --reverse a73c601e6..224b73e24); do
  s=$(git log -1 --format=%s $c)
  grep -qxF "$s" <<< "$done_subjects" && continue
  if git -c rerere.enabled=false cherry-pick $c > /root/rebase-v110/cp-$(git rev-parse --short $c).out 2>&1; then
    echo "ok   $(git rev-parse --short $c) ${s:0:80}"
  else
    echo "STOP $(git rev-parse --short $c) ${s:0:80}"
    for f in $(git diff --name-only --diff-filter=U); do echo "  $f: $(grep -c '^<<<<<<<' $f) hunks"; done
    exit 2
  fi
done
echo "ALL PICKED: $(git rev-list --count merge-v1.1.0..HEAD) commits on merge-v1.1.0"
