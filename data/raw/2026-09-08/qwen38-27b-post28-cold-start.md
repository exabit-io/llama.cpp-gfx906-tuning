# qwen38-27b-post28 cold start 2026-09-08T21:55:54+00:00: fresh llama-server (-np 8 -c 65536) x4 per build; load-to-ready, then the client's steps (seq1 = 2K prefill + 64 tokens, seq2-6 = decode-only 64-token continuations, batch8 = 8 x 1300-token prompts + 64 tokens)
| process | step | s since ready | prompt n | prompt ms | gen n | gen t/s |
|---|---|---:|---:|---:|---:|---:|
| prod-p1 | load-to-ready | 19.3 | | | | |
| prod-p1 | seq1 (cum 64 tok) | 3.4 | 2048 | 2037 | 64 | 53.0 |
| prod-p1 | seq2 (cum 128 tok) | 4.7 | 4 | 225 | 64 | 56.4 |
| prod-p1 | seq3 (cum 192 tok) | 6.1 | 4 | 230 | 64 | 56.2 |
| prod-p1 | seq4 (cum 256 tok) | 7.5 | 4 | 233 | 64 | 56.5 |
| prod-p1 | seq5 (cum 320 tok) | 8.9 | 4 | 230 | 64 | 56.0 |
| prod-p1 | seq6 (cum 384 tok) | 10.2 | 4 | 231 | 64 | 56.1 |
| prod-p1 | batch8 #1 | 26.5 | 1300x8 | 7082 | 512 | 12.8 per req, 102.4 agg |
| prod-p1 | batch8 #2 | 44.3 | 1300x8 | 7055 | 512 | 13.0 per req, 103.8 agg |
| r2-nr0-p1 | load-to-ready | 11.3 | | | | |
| r2-nr0-p1 | seq1 (cum 64 tok) | 3.2 | 2048 | 1818 | 64 | 50.0 |
| r2-nr0-p1 | seq2 (cum 128 tok) | 4.7 | 4 | 259 | 64 | 50.8 |
| r2-nr0-p1 | seq3 (cum 192 tok) | 6.2 | 4 | 234 | 64 | 54.6 |
| r2-nr0-p1 | seq4 (cum 256 tok) | 7.6 | 4 | 235 | 64 | 54.4 |
| r2-nr0-p1 | seq5 (cum 320 tok) | 9.0 | 4 | 235 | 64 | 54.1 |
| r2-nr0-p1 | seq6 (cum 384 tok) | 10.4 | 4 | 236 | 64 | 54.2 |
| r2-nr0-p1 | batch8 #1 | 25.8 | 1300x8 | 6522 | 512 | 12.6 per req, 100.7 agg |
| r2-nr0-p1 | batch8 #2 | 42.9 | 1300x8 | 6696 | 512 | 12.7 per req, 101.3 agg |
| r2-nr1-p1 | load-to-ready | 20.8 | | | | |
| r2-nr1-p1 | seq1 (cum 64 tok) | 3.6 | 2048 | 2239 | 64 | 50.6 |
| r2-nr1-p1 | seq2 (cum 128 tok) | 5.1 | 4 | 258 | 64 | 52.2 |
| r2-nr1-p1 | seq3 (cum 192 tok) | 6.5 | 4 | 236 | 64 | 54.9 |
| r2-nr1-p1 | seq4 (cum 256 tok) | 7.9 | 4 | 234 | 64 | 54.9 |
| r2-nr1-p1 | seq5 (cum 320 tok) | 9.3 | 4 | 234 | 64 | 54.5 |
| r2-nr1-p1 | seq6 (cum 384 tok) | 10.8 | 4 | 241 | 64 | 54.5 |
| r2-nr1-p1 | batch8 #1 | 27.8 | 1300x8 | 7463 | 512 | 12.4 per req, 99.3 agg |
| r2-nr1-p1 | batch8 #2 | 46.5 | 1300x8 | 7598 | 512 | 12.4 per req, 99.2 agg |
| r2-nr0-p2 | load-to-ready | 23.6 | | | | |
| r2-nr0-p2 | seq1 (cum 64 tok) | 5.4 | 2048 | 2812 | 64 | 26.3 |
| r2-nr0-p2 | seq2 (cum 128 tok) | 8.1 | 4 | 260 | 64 | 25.9 |
| r2-nr0-p2 | seq3 (cum 192 tok) | 9.5 | 4 | 237 | 64 | 54.1 |
| r2-nr0-p2 | seq4 (cum 256 tok) | 11.0 | 4 | 235 | 64 | 54.1 |
| r2-nr0-p2 | seq5 (cum 320 tok) | 12.4 | 4 | 237 | 64 | 53.7 |
| r2-nr0-p2 | seq6 (cum 384 tok) | 13.8 | 4 | 235 | 64 | 53.8 |
| r2-nr0-p2 | batch8 #1 | 31.3 | 1300x8 | 7023 | 512 | 12.4 per req, 99.4 agg |
| r2-nr0-p2 | batch8 #2 | 48.5 | 1300x8 | 6671 | 512 | 12.7 per req, 101.4 agg |
| r2-nr1-p2 | load-to-ready | 20.4 | | | | |
| r2-nr1-p2 | seq1 (cum 64 tok) | 3.6 | 2048 | 2261 | 64 | 51.8 |
| r2-nr1-p2 | seq2 (cum 128 tok) | 5.1 | 4 | 256 | 64 | 52.1 |
| r2-nr1-p2 | seq3 (cum 192 tok) | 6.5 | 4 | 232 | 64 | 55.2 |
| r2-nr1-p2 | seq4 (cum 256 tok) | 7.9 | 4 | 231 | 64 | 55.3 |
| r2-nr1-p2 | seq5 (cum 320 tok) | 9.3 | 4 | 235 | 64 | 54.8 |
| r2-nr1-p2 | seq6 (cum 384 tok) | 10.8 | 4 | 271 | 64 | 54.8 |
| r2-nr1-p2 | batch8 #1 | 27.6 | 1300x8 | 7417 | 512 | 12.4 per req, 99.3 agg |
| r2-nr1-p2 | batch8 #2 | 46.4 | 1300x8 | 7651 | 512 | 12.4 per req, 99.4 agg |
| prod-p2 | load-to-ready | 19.5 | | | | |
| prod-p2 | seq1 (cum 64 tok) | 3.4 | 2048 | 2040 | 64 | 52.9 |
| prod-p2 | seq2 (cum 128 tok) | 4.7 | 4 | 226 | 64 | 56.4 |
| prod-p2 | seq3 (cum 192 tok) | 6.1 | 4 | 229 | 64 | 56.3 |
| prod-p2 | seq4 (cum 256 tok) | 7.5 | 4 | 230 | 64 | 56.6 |
| prod-p2 | seq5 (cum 320 tok) | 8.9 | 4 | 231 | 64 | 56.1 |
| prod-p2 | seq6 (cum 384 tok) | 10.2 | 4 | 233 | 64 | 56.0 |
| prod-p2 | batch8 #1 | 26.5 | 1300x8 | 7070 | 512 | 12.8 per req, 102.8 agg |
| prod-p2 | batch8 #2 | 44.3 | 1300x8 | 7058 | 512 | 13.0 per req, 103.8 agg |
| r2-nr1-p3 | load-to-ready | 19.6 | | | | |
| r2-nr1-p3 | seq1 (cum 64 tok) | 3.5 | 2048 | 2095 | 64 | 51.6 |
| r2-nr1-p3 | seq2 (cum 128 tok) | 5.0 | 4 | 258 | 64 | 51.5 |
| r2-nr1-p3 | seq3 (cum 192 tok) | 6.4 | 4 | 231 | 64 | 55.0 |
| r2-nr1-p3 | seq4 (cum 256 tok) | 7.8 | 4 | 233 | 64 | 54.9 |
| r2-nr1-p3 | seq5 (cum 320 tok) | 9.2 | 4 | 232 | 64 | 54.5 |
| r2-nr1-p3 | seq6 (cum 384 tok) | 10.6 | 4 | 272 | 64 | 54.5 |
| r2-nr1-p3 | batch8 #1 | 29.0 | 1300x8 | 8139 | 512 | 12.2 per req, 97.5 agg |
| r2-nr1-p3 | batch8 #2 | 47.8 | 1300x8 | 7676 | 512 | 12.4 per req, 98.9 agg |
| prod-p3 | load-to-ready | 18.6 | | | | |
| prod-p3 | seq1 (cum 64 tok) | 3.4 | 2048 | 2029 | 64 | 53.0 |
| prod-p3 | seq2 (cum 128 tok) | 4.7 | 4 | 225 | 64 | 56.4 |
| prod-p3 | seq3 (cum 192 tok) | 6.1 | 4 | 231 | 64 | 56.3 |
| prod-p3 | seq4 (cum 256 tok) | 7.5 | 4 | 232 | 64 | 56.5 |
| prod-p3 | seq5 (cum 320 tok) | 8.8 | 4 | 235 | 64 | 56.1 |
| prod-p3 | seq6 (cum 384 tok) | 10.2 | 4 | 232 | 64 | 56.0 |
| prod-p3 | batch8 #1 | 26.4 | 1300x8 | 7075 | 512 | 12.8 per req, 102.7 agg |
| prod-p3 | batch8 #2 | 44.4 | 1300x8 | 7057 | 512 | 12.9 per req, 103.6 agg |
| r2-nr0-p3 | load-to-ready | 25.6 | | | | |
| r2-nr0-p3 | seq1 (cum 64 tok) | 5.3 | 2048 | 2729 | 64 | 25.9 |
| r2-nr0-p3 | seq2 (cum 128 tok) | 7.1 | 4 | 256 | 64 | 43.1 |
| r2-nr0-p3 | seq3 (cum 192 tok) | 8.5 | 4 | 234 | 64 | 54.3 |
| r2-nr0-p3 | seq4 (cum 256 tok) | 9.9 | 4 | 236 | 64 | 54.5 |
| r2-nr0-p3 | seq5 (cum 320 tok) | 11.3 | 4 | 235 | 64 | 54.2 |
| r2-nr0-p3 | seq6 (cum 384 tok) | 12.7 | 4 | 235 | 64 | 54.2 |
| r2-nr0-p3 | batch8 #1 | 30.2 | 1300x8 | 7243 | 512 | 12.5 per req, 100.0 agg |
| r2-nr0-p3 | batch8 #2 | 47.5 | 1300x8 | 6774 | 512 | 12.7 per req, 101.6 agg |
| prod-p4 | load-to-ready | 22.6 | | | | |
| prod-p4 | seq1 (cum 64 tok) | 3.4 | 2048 | 2067 | 64 | 53.0 |
| prod-p4 | seq2 (cum 128 tok) | 4.8 | 4 | 225 | 64 | 56.6 |
| prod-p4 | seq3 (cum 192 tok) | 6.1 | 4 | 230 | 64 | 56.4 |
| prod-p4 | seq4 (cum 256 tok) | 7.5 | 4 | 230 | 64 | 56.6 |
| prod-p4 | seq5 (cum 320 tok) | 8.9 | 4 | 231 | 64 | 56.1 |
| prod-p4 | seq6 (cum 384 tok) | 10.3 | 4 | 235 | 64 | 56.1 |
| prod-p4 | batch8 #1 | 26.4 | 1300x8 | 7075 | 512 | 12.8 per req, 102.6 agg |
| prod-p4 | batch8 #2 | 44.3 | 1300x8 | 7058 | 512 | 12.9 per req, 103.5 agg |
| r2-nr1-p4 | load-to-ready | 19.7 | | | | |
| r2-nr1-p4 | seq1 (cum 64 tok) | 3.5 | 2048 | 2097 | 64 | 50.6 |
| r2-nr1-p4 | seq2 (cum 128 tok) | 5.0 | 4 | 255 | 64 | 50.6 |
| r2-nr1-p4 | seq3 (cum 192 tok) | 6.4 | 4 | 233 | 64 | 54.9 |
| r2-nr1-p4 | seq4 (cum 256 tok) | 7.8 | 4 | 234 | 64 | 55.0 |
| r2-nr1-p4 | seq5 (cum 320 tok) | 9.2 | 4 | 237 | 64 | 54.6 |
| r2-nr1-p4 | seq6 (cum 384 tok) | 10.7 | 4 | 273 | 64 | 54.6 |
| r2-nr1-p4 | batch8 #1 | 29.1 | 1300x8 | 8177 | 512 | 12.2 per req, 97.2 agg |
| r2-nr1-p4 | batch8 #2 | 48.1 | 1300x8 | 7702 | 512 | 12.4 per req, 98.8 agg |
| r2-nr0-p4 | load-to-ready | 20.6 | | | | |
| r2-nr0-p4 | seq1 (cum 64 tok) | 4.6 | 2048 | 2339 | 64 | 29.8 |
| r2-nr0-p4 | seq2 (cum 128 tok) | 6.4 | 4 | 258 | 64 | 41.6 |
| r2-nr0-p4 | seq3 (cum 192 tok) | 7.8 | 4 | 236 | 64 | 54.2 |
| r2-nr0-p4 | seq4 (cum 256 tok) | 9.3 | 4 | 238 | 64 | 54.3 |
| r2-nr0-p4 | seq5 (cum 320 tok) | 10.7 | 4 | 235 | 64 | 53.9 |
| r2-nr0-p4 | seq6 (cum 384 tok) | 12.1 | 4 | 235 | 64 | 53.9 |
| r2-nr0-p4 | batch8 #1 | 30.3 | 1300x8 | 7306 | 512 | 12.5 per req, 99.8 agg |
| r2-nr0-p4 | batch8 #2 | 47.6 | 1300x8 | 6793 | 512 | 12.7 per req, 101.4 agg |
# done 2026-09-08T22:10:00+00:00
