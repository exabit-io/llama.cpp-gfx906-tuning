#!/bin/bash
# push-v110.sh -- the v1.1.0 branches and tags of exabit-io/mx-llama.cpp. Run ONLY after the release queue's rules
# have passed (README, section on the release queue). Nothing here touches mxxm-t's or upstream's repositories.
#   master-v110      -> master, gfx906-single, gfx906-multi   (forced with a lease on the v1.0.0 r2 master 224b73e24)
#   candidates-v110  -> gfx906-candidates                     (forced with a lease on 5c01f4ffd)
#   annotated tags gfx906/v1.1.0/r0/merge (merge-v1.1.0), /master, /candidates; messages = tagmsg-*.txt, no '@@' left
#   the wip/* branches of this cycle are deleted on the remote afterwards (wip/master-v110 = 9a7177d0a has the tree of
#   master~4 = e2497e7c0; wip/candidates-v110 = 8eb35b2d6 is the parent of candidates-v110; wip/ds41-upstream-layout
#   is the DeepSeek commit before its port note)
# DRY=1 (the default) checks everything and prints the commands; DRY=0 runs them.
set -u
G=/root/exabit-llama.cpp; W=/root/rebase-v110; REM=exabit-mx; DRY=${DRY:-1}
export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519_exabit -o IdentitiesOnly=yes"
M=c782c317dcd40170d3abc51d58c6263663a08fdc; OLDM=224b73e24; OLDC=5c01f4ffd
die(){ echo "FATAL: $*"; exit 1; }
run(){ if [ "$DRY" = 1 ]; then echo "DRY: $*"; else echo "+ $*"; "$@" || die "failed: $*"; fi; }
cd $G || exit 1
[ "$(git rev-parse master-v110)" = $M ] || die "master-v110 is not $M"
C=$(git rev-parse candidates-v110); MG=$(git rev-parse merge-v1.1.0)
[ "$(git config user.email)" = joshua@exabit.io ] || die "git identity is not Exabit's"
for b in master-v110 candidates-v110; do
  n=$(git log --format='%ae %ce' merge-v1.1.0..$b | grep -vc '^joshua@exabit.io joshua@exabit.io$'); [ "$n" = 0 ] || die "$b has $n commits with another identity"
done
git merge-base --is-ancestor $MG master-v110 || die "merge-v1.1.0 is not below master-v110"
git merge-base --is-ancestor $MG candidates-v110 || die "merge-v1.1.0 is not below candidates-v110"
DEB=$(cat /root/mx-llama-deb/last-deb.txt); [ "$(dpkg-deb --fsys-tarfile "$DEB" | tar -xO ./usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT)" = $M ] || die "the deb is not built from $M"
for t in merge master candidates; do
  [ -s $W/tagmsg-$t.txt ] || die "no tagmsg-$t.txt"; grep -q '@@' $W/tagmsg-$t.txt && die "tagmsg-$t.txt has a placeholder"
done
rem=$(git ls-remote $REM 'refs/heads/*' 'refs/tags/gfx906/v1.1.0/*') || die "ls-remote"
rhead(){ echo "$rem" | awk -v r="refs/heads/$1" '$2 == r { print $1 }'; }
echo "remote now: master $(rhead master | cut -c1-9) single $(rhead gfx906-single | cut -c1-9) multi $(rhead gfx906-multi | cut -c1-9) candidates $(rhead gfx906-candidates | cut -c1-9) merge-v1.1.0 $(rhead merge-v1.1.0 | cut -c1-9)"
for b in master gfx906-single gfx906-multi; do case "$(rhead $b)" in $OLDM*|$M) ;; *) die "remote $b is $(rhead $b), neither $OLDM nor the new master";; esac; done
case "$(rhead gfx906-candidates)" in $OLDC*|$C) ;; *) die "remote gfx906-candidates is $(rhead gfx906-candidates)";; esac
[ "$(rhead merge-v1.1.0)" = $MG ] || die "remote merge-v1.1.0 is not $MG"
echo "$rem" | grep -q 'refs/tags/gfx906/v1.1.0/' && echo "NOTE: v1.1.0 tags already on the remote: $(echo "$rem" | grep -c 'refs/tags/gfx906/v1.1.0/')"
OM=$(rhead master); OS=$(rhead gfx906-single); OU=$(rhead gfx906-multi); OC=$(rhead gfx906-candidates)
run git push --force-with-lease=master:$OM --force-with-lease=gfx906-single:$OS --force-with-lease=gfx906-multi:$OU $REM master-v110:master master-v110:gfx906-single master-v110:gfx906-multi
run git push --force-with-lease=gfx906-candidates:$OC $REM candidates-v110:gfx906-candidates
tagit(){ # tagit NAME COMMIT MSGFILE
  if git rev-parse -q --verify "refs/tags/$1" >/dev/null; then [ "$(git rev-parse "$1^{commit}")" = "$2" ] || die "local tag $1 is on another commit"; echo "tag $1 exists"
  else run git tag -a "$1" "$2" -F "$3"; fi
}
tagit gfx906/v1.1.0/r0/merge $MG $W/tagmsg-merge.txt
tagit gfx906/v1.1.0/r0/master $M $W/tagmsg-master.txt
tagit gfx906/v1.1.0/r0/candidates $C $W/tagmsg-candidates.txt
run git push $REM gfx906/v1.1.0/r0/merge gfx906/v1.1.0/r0/master gfx906/v1.1.0/r0/candidates
wip=$(echo "$rem" | awk '$2 ~ /^refs\/heads\/wip\// { sub("refs/heads/", "", $2); print $2 }' | grep -E '^wip/(master-v110|candidates-v110|ds41-upstream-layout)$')
[ -n "$wip" ] && run git push $REM --delete $wip
if [ "$DRY" != 1 ]; then
  git fetch -q $REM
  echo "remote after: master $(git rev-parse --short $REM/master) single $(git rev-parse --short $REM/gfx906-single) multi $(git rev-parse --short $REM/gfx906-multi) candidates $(git rev-parse --short $REM/gfx906-candidates)"
  git ls-remote $REM 'refs/tags/gfx906/v1.1.0/*' | cut -c1-9,41-
fi
echo "=== PUSH-V110 $( [ "$DRY" = 1 ] && echo DRY || echo DONE ) ==="
