# post25b 2026-09-08T20:03:59+00:00: branch r2 --no-repack, batched 8/12/16 at 2K with the CUDA op fusion on and off; production 175 / 197.5 / 202.9
| build | fusion | 8 | 12 | 16 |
|---|---|---:|---:|---:|
| r2-nr1-fusion-on | 125.58 | 184.77 | 194.34 | 
| r2-nr1-fusion-off | 147.72 | 187.13 | 187.11 | 
| r2-nr1-no-custom-ar | 153.30 | 187.66 | 193.87 | 
| prod-fusion-on | 175.10 | 197.39 | 202.92 | 
| prod-fusion-off | 169.21 | 190.75 | 196.15 | 

## reverse order -npl 16,12,8: is the 8-slot loss the after-load warm-up on the first cell?
| build | fusion | 16 | 12 | 8 |
|---|---|---:|---:|---:|
| r2-nr1-rev | 187.33 | 194.82 | 142.65 | 
| prod-rev | 203.50 | 197.49 | 174.63 | 
| r2-nr0-rev | 144.86 | 136.21 | 150.37 | 
# done 2026-09-08T20:20:12+00:00
