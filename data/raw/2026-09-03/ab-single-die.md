# single-die server A/B, Q4_K_M, conc 1, prompt 1300 gen 256; llama-bench ref on rocm0: pp2048 202 t/s, tg256 24.7 t/s  2026-09-03T16:00:16+00:00
| variant | args | per-req gen t/s | TTFT s (1300 tok) |
|---|---|---:|---:|
| A-baseline | --device rocm0 -sm none -ngl all -fit off -fa on -np 8 -cb -c 32768 -b 2048 -ub 2048 --host 127.0.0.1 --port 8089 |  17.1 | 10.80  |
| B-no-cache-ram | --device rocm0 -sm none -ngl all -fit off -fa on -np 8 -cb -c 32768 -b 2048 -ub 2048 --host 127.0.0.1 --port 8089 --cache-ram 0 |  17.2 | 10.80  |
| C-np1 | --device rocm0 -sm none -ngl all -fit off -fa on -np 1 -c 4096 -b 2048 -ub 2048 --host 127.0.0.1 --port 8089 |  17.2 | 10.81  |
| D-sm-tensor | --device rocm0 -sm tensor -ngl all -fit off -fa on -np 8 -cb -c 32768 -b 2048 -ub 2048 --host 127.0.0.1 --port 8089 |  17.0 | 10.86  |
| E-nodev-mg0 | -sm none -mg 0 -ngl all -fit off -fa on -np 8 -cb -c 32768 -b 2048 -ub 2048 --host 127.0.0.1 --port 8089 |  17.1 | 10.80  |
| F-fa-off | --device rocm0 -sm none -ngl all -fit off -fa off -np 8 -cb -c 32768 -b 2048 -ub 2048 --host 127.0.0.1 --port 8089 |  16.5 | 10.95  |
| G-threads4 | --device rocm0 -sm none -ngl all -fit off -fa on -np 8 -cb -c 32768 -b 2048 -ub 2048 --host 127.0.0.1 --port 8089 -t 4 -tb 4 |  17.2 | 10.80  |
| H-np1-faoff-nocache | --device rocm0 -sm none -ngl all -fit off -fa off -np 1 -c 4096 -b 2048 -ub 2048 --host 127.0.0.1 --port 8089 --cache-ram 0 |  16.5 | 10.96  |
# done 2026-09-03T16:15:19+00:00
