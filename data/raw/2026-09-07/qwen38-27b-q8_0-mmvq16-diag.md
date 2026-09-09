# mmvq16 diagnostics  2026-09-07T10:43:44+00:00

## kernel names containing mul_mat during batch-12 decode, /opt/llama.cpp-mmvq16 (counts from rocprofv3 kernel trace)
    8451  void mul_mat_vec_q<
    1488  void mul_mat_q<

## kernel names containing mul_mat during batch-12 decode, /opt/llama.cpp (counts from rocprofv3 kernel trace)
    9168  void mul_mat_q<
     771  void mul_mat_vec_q<
