#pragma once

void matmul_initialize(const float *A, const float *B);

void matmul_cublas();
void matmul_naive();
void matmul_global_mem_coalescing();
void matmul_shared_mem();
void matmul_1d_blocktiling_naive();
void matmul_1d_blocktiling_row();
void matmul_1d_blocktiling_row_vector_type();
void matmul_1d_blocktiling_column();
void matmul_2d_blocktiling();
void matmul_2d_blocktiling_loop_unrolling();

void matmul_finalize(const float *C);