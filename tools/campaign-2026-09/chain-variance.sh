#!/bin/bash
set -u; W=/root/night-20260919
/bin/bash $W/waitproc.sh "${1:-}" "^[^ ]*/bin/llama-perplexity" || echo "$(date -Is) waitproc failed ($?)" >> $W/chain.log
echo "$(date -Is) perplexity finished; starting variance gate (Part 6 gate 3)" >> $W/chain.log
sleep 20
/bin/bash $W/variance-gate.sh >> $W/variance.out 2>&1
echo "$(date -Is) variance gate exit=$?" >> $W/chain.log
echo "$(date -Is) === VARIANCE GATE DONE ===" >> $W/chain.log
