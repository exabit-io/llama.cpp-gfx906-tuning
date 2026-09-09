# power-draw-test  result=clamp3

| phase | what | s | SMC DC total max W | SMC AC est. max W | bay1 / bay2 max W | CPU zone max W | 4-die sum max W (sampler) | max single die W | zone T (0/3/4/7) max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | idle baseline 60 s | 60 | 302 | 358 | 105 / 101 | 37 | 168 | 44 | 0.0/0.0/0.0/0.0 |
| 1 | rocBLAS (FORCE_CUBLAS) single-die pp2048 on rocm0, -r 4, the known trigger | 64 | 548 | 675 | 296 / 98 | 84 | 341 | 223 | 0.0/0.0/0.0/0.0 |
| 2 | stock tp4 prefill -p 8192 -ub 2048 -r 6, CPU idle (four dies at the cap) | 90 | 1062 | 1353 | 460 / 453 | 88 | 784 | 216 | 0.0/0.0/0.0/0.0 |
| 3 | same tp4 prefill + CPU loaded on all 56 threads (openssl speed aes-256-gcm) | 143 | 1264 | 1428 | 459 / 451 | 303 | 802 | 218 | 59.1/0.0/0.0/0.0 |

Pair each phase with the meter's peak register (clear it before the run). SMC AC estimate runs ~40 W above the meter at idle; the DC envelope is 1228 W, bays 530 W each, AC 1562 W.

## SMC key diff, clean vs clamped (2026-09-07 10:40 UTC)

`smc-keys-clean-0907-0957.txt` (cold-cycled, unclamped, taken by queue-after-reboot.sh) against `smc-keys-clamped-0813.txt` and
`smc-keys-clamped-0920.txt`. 1282 decoded keys each. Keys equal in both clamped dumps and different in the clean one: 25, of which 24 are
sensors (TC1P/TC1p CPU proximity, TG2d/TG5d/TJ2d/TJ5d GPU die temps, F*Mx/F*Td/F*Te fan targets, ADC7/ADCk, AC-M, FRab/FRac, PfG4, SPHS).
The only non-sensor key is `CLWK` (ui16): 65535 in both clamped dumps, 27240 clean. It is not a latch indicator: read at 10:40 on the
unclamped box under a four-die load it is also 65535 (the clean dump was taken at idle), so it tracks load. `TG1s` and `TG4s` read 1
only in the 0813 dump (0 at 0920 and clean). Conclusion: no key in a static dump distinguishes the clamped state; the SMC's latch is
not exposed as a readable key value, and the DC total (`PZ0G`) against `PZ0E` remains the only usable signal. Method: parse
"name 'type' len hex decoded" lines, compare hex fields.
