// Copyright (c) 2023-present, Junyeol Ryu

#pragma once

__global__ void matmul_naive_kernel(float *d_M, float *d_N, float *d_P, int M, int N, int K);

__global__ void matmul_global_mem_coalescing_kernel(float *d_M, float *d_N, float *d_P, int M, int N, int K);

#define SHARED_MEM_TILE_SIZE 32
__global__ void matmul_shared_mem_kernel(float *d_M, float *d_N, float *d_P, int M, int N, int K);

#define ONE_D_BLOCKTILING_NAIVE_TILE_SIZE 64
#define ONE_D_BLOCKTILING_NAIVE_WPT 4 // Intentionally set to maximize thread block size, which is 
                                // ONE_D_BLOCKTILING_NAIVE_TILE_SIZE * ONE_D_BLOCKTILING_NAIVE_TILE_SIZE / ONE_D_BLOCKTILING_NAIVE_WPT
__global__ void matmul_1d_blocktiling_naive_kernel(float *d_M, float *d_N, float *d_P, int M, int N, int K);

#define ONE_D_BLOCKTILING_ROW_TILE_SIZE 64
#define ONE_D_BLOCKTILING_ROW_WPT 4
__global__ void matmul_1d_blocktiling_row_kernel(float *d_M, float *d_N, float *d_P, int M, int N, int K);

#define ONE_D_BLOCKTILING_ROW_VECTOR_TYPE_TILE_SIZE 64
#define ONE_D_BLOCKTILING_ROW_VECTOR_TYPE_WPT 4
__global__ void matmul_1d_blocktiling_row_vector_type_kernel(float4 *d_M, float4 *d_N, float4 *d_P, int M, int N, int K);

#define ONE_D_BLOCKTILING_COL_TILE_SIZE 64
#define ONE_D_BLOCKTILING_COL_WPT 4
__global__ void matmul_1d_blocktiling_column_kernel(float *d_M, float *d_N, float *d_P, int M, int N, int K);

#define TWO_D_BLOCKTILING_TILE_SIZE 64 // Must be a multiple of `TWO_D_BLOCKTILING_CPT * TWO_D_BLOCKTILING_RPT`
#define TWO_D_BLOCKTILING_CPT 4 // Columns per thread
#define TWO_D_BLOCKTILING_RPT 4 // Rows per thread
__global__ void matmul_2d_blocktiling_kernel(float *d_M, float *d_N, float *d_P, int M, int N, int K);

#define TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE 64 // Must be a multiple of `TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT * TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT`
#define TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT 4 // Columns per thread
#define TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT 4 // Rows per thread
__global__ void matmul_2d_blocktiling_loop_unrolling_kernel(float *d_M, float *d_N, float *d_P, int M, int N, int K);