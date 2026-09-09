#!/bin/bash
# 5 s log of the SMC's own power view: zone readings (G) and the zeroed fourth counter (T) for DC total (0), CPU (1), MPX bays (3,4),
# PCIe (5) and AC input (7) plus the filtered values (F) of zones 0/1/3/4; GPU rail powers; the per-die telemetry status flags (s) and die temps (D). Envelopes (E) at idle 2026-09-07:
# PZ0E 1228  PZ1E 450  PZ3E 530  PZ4E 530  PZ5E 300  PZ7E 1562 (see smc-keys-clamped-0813.txt).   usage: nohup smc-log.sh FILE &
F=${1:-/root/rocm-tests/bench/smc-power.log}; B=/root/rocm-tests/bench
echo "# $(date -Is) smc-log start; envelopes: $($B/smc-read.py PZ0E PZ1E PZ3E PZ4E PZ5E PZ7E)" >> $F
while true; do echo "$(date +%T) $($B/smc-read.py PZ0G PZ7G PZ1G PZ3G PZ4G PZ5G PZ0T PZ3T PZ4T PZ7T PZ0F PZ1F PZ3F PZ4F PG0R PG2R TG0s TG1s TG3s TG4s TG0D TG1D TG3D TG4D 2>&1)" >> $F; sleep 5; done
