#!/bin/bash
# queue1.sh -- dio check (4 builds), fold check (reconciled exabit master), PR suite (v0.7.0 merge)
set -u
W=/root/rebase-v070
echo "$(date -Is) queue1 start pid $$" >> $W/queue1.log
$W/dio-check.sh $W/dio pre-v050=/opt/llama.cpp-mxxm-fh-rocm10 pr17-v050=/root/build-mx-sync reconcile-b11553=/root/build-reconcile merge-v070=/root/build-merge-v070
echo "$(date -Is) dio rc=$?" >> $W/queue1.log
$W/fold-check.sh /root/build-reconcile $W/fold-reconcile
echo "$(date -Is) fold rc=$?" >> $W/queue1.log
SRC=/root/wt-merge-v070 $W/pr-test.sh /root/build-merge-v070 $W/pr-test-merge
echo "$(date -Is) pr-test rc=$?" >> $W/queue1.log
echo "$(date -Is) queue1 done" >> $W/queue1.log
