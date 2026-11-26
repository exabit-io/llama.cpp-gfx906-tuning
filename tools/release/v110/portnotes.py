#!/usr/bin/env python3
# msg-filter: append the port note for terms adapted to mxxm-t c4d12e2c0 (upstream b11553)
import sys
msg = sys.stdin.read()
head = "Ported onto mxxm-t/mx-llama.cpp c4d12e2c0 (upstream b11553): "
notes = {
 "term 02-": "rms_norm_f32 now walks clamped channel/sample grids and has do_scale; do_q8 is the last template flag and the q8 row offset is taken per channel inside the loop.",
 "term 03-": "the preadd input/output rows are offset per (sample, channel) inside the clamped-grid loop; the extended launcher passes the clamped flag through.",
 "term 04-": "the recurrent GDN kernel now splits a warp into gdn_cols_per_warp column segments that each hold the whole head vector, so the folded L2 norm reduces over lanes_per_col lanes (a warp-wide sum counts the vector cols_per_warp times). The chunked kernel does not apply the folds, so a folded call stays on the recurrent kernel (reachable for KDA, chunk 16, at 32-64 tokens).",
 "term 10-": "gdn_l2_sumsq takes lanes_per_col instead of the warp size. With 16 lanes per column (64-lane warp, 4 columns) each lane runs the even and odd l2_norm_f32 chains and combines them in the order of the warp reduction in use (generic: offset 16 first; GCN DPP: offset 1 first).",
 "term 12-": "uses the lanes_per_col form of gdn_l2_sumsq.",
 "term 23-": "keeps upstream's !defined(GGML_USE_MUSA) on the Ampere __reduce_add_sync branch.",
 "term 01-": "the MUL_MAT_ID MMVQ window keeps upstream's Pascal and MUSA conditions and returns MMVQ_MMID_MAX_BATCH_SIZE.",
}
for k, v in notes.items():
    if msg.startswith(k):
        msg = msg.rstrip("\n") + "\n\n" + head + v + "\n"
        break
sys.stdout.write(msg)
