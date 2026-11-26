#!/usr/bin/env python3
# msg-filter: append the port note of the v1.1.0 cycle to the Exabit commits that had to be adapted
# (git filter-branch --msg-filter 'python3 /root/rebase-v110/portnotes-v110.py' merge-v1.1.0..master-v110)
import sys
msg = sys.stdin.read()
head = "Rebased onto mxxm-t/mx-llama.cpp 557a6068f + llama.cpp v1.1.0 (b12774): "
notes = {
 "term 02-": "rms_norm_f32 is a template over float and BF16 rows (#32135) and the launcher takes the clamped grid as a tag. do_q8 is the last template switch after T_add and clamped, is asserted for float rows, and the launcher instantiates it only for float (if constexpr).",
 "term 03-": "do_preadd follows do_q8 and is float only as well. The launcher chooses the clamped grid, the q8 output and the pre-add with three nested tags and keeps upstream's branch for a bias of the row type.",
 "term 04-": "the recurrent kernel also takes upstream's replay arguments (#31669). The column-thread and chunked kernels take a call only when it has neither replay inputs nor folded producers; an op with replay inputs is never folded (the kernel asserts it, the next but one commit keeps the walk off such ops).",
 "term 10-": "gdn_l2_sumsq is defined before the kernel, after upstream's gdn_replay_args.",
 "term 23-": "the i32 reduction keeps upstream's GGML_CUDA_FULL_THREAD_MASK on the Ampere branch, and the 64-lane steps use ggml_cuda_shfl_xor: the raw _sync primitives take a 64-bit mask on HIP since #31617.",
 "term 01-": "the MUL_MAT_ID window keeps upstream's few-row MMQ (use_mmq_few, #32065) in front of MMVQ_MMID_MAX_BATCH_SIZE. calc_rows_per_block keeps upstream's RDNA1 table. The two unreferenced files this commit used to carry (common/speculative-adaptive.h, tests/test-speculative-adaptive.cpp) are dropped.",
 "term 05-": "the GCN rows-per-block return sits before upstream's GENERIC / RDNA1 condition.",
}
for k, v in notes.items():
    if msg.startswith(k) and head not in msg:
        msg = msg.rstrip("\n") + "\n\n" + head + v + "\n"
        break
sys.stdout.write(msg)
