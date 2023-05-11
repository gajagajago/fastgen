// Copyright (c) 2023-present, Junyeol Ryu

#include "matmul.h"
#include "util.h"
#include "matmul-kernel.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// #define DEBUG1   // Comment this line to disable LOG1 & LOG2
// #define DEBUG2   // Comment this line to disable LOG2

#ifdef DEBUG1
#define LOG1 1
#else
#define LOG1 0
#endif 

#if defined DEBUG1 && defined DEBUG2
#define LOG2 1
#else
#define LOG2 0
#endif 

#ifndef CUBLAS_CALL
#define CUBLAS_CALL(f)                                      \
  {                                                         \
    cublasStatus_t err = (f);                               \
    if (err != CUBLAS_STATUS_SUCCESS) {                     \
      fprintf(stderr, "cuBLAS error at [%s:%d] %d %s\n",    \
               __FILE__, __LINE__,                          \
              err, cublasGetStatusString(err));             \
      exit(1);                                              \
    }                                                       \
  }
#endif
cublasHandle_t handle;

#ifndef CUDA_CALL
#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n",     \
               __FILE__, __LINE__,                                             \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }
#endif

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

#ifndef MAX
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#endif

float *d_A, *d_B, *d_C;
extern int M, N, K;

void matmul_initialize(const float *A, const float* B)
{
  CUDA_CALL(cudaSetDevice(0));
  CUBLAS_CALL(cublasCreate(&handle));

  CUDA_CALL(cudaMalloc(&d_A, M * K * sizeof(float)));
  CUDA_CALL(cudaMalloc(&d_B, K * N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&d_C, M * N * sizeof(float)));

  CUDA_CALL(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
}

void matmul_finalize(const float* C)
{
  CUDA_CALL(cudaSetDevice(0));
  CUDA_CALL(cudaMemcpy((void*)C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(d_A));
  CUDA_CALL(cudaFree(d_B));
  CUDA_CALL(cudaFree(d_C));
    
  CUBLAS_CALL(cublasDestroy(handle));
}


void matmul_naive() 
{
  // DO NOT REMOVE
  CUDA_CALL(cudaSetDevice(0));
  
  dim3 dimGrid(N/32, M/32);
  dim3 dimBlock(32, 32);
  matmul_naive_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

  // DO NOT REMOVE
  CUDA_CALL(cudaDeviceSynchronize());
}

void matmul_global_mem_coalescing() 
{
  // DO NOT REMOVE
  CUDA_CALL(cudaSetDevice(0));
  
  dim3 dimGrid(N/32, M/32);
  dim3 dimBlock(32, 32);
  matmul_global_mem_coalescing_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

  // DO NOT REMOVE
  CUDA_CALL(cudaDeviceSynchronize());
}

void matmul_shared_mem() 
{
  // DO NOT REMOVE
  CUDA_CALL(cudaSetDevice(0));

  dim3 dimGrid(N / SHARED_MEM_TILE_SIZE, M / SHARED_MEM_TILE_SIZE);
  dim3 dimBlock(SHARED_MEM_TILE_SIZE, SHARED_MEM_TILE_SIZE);   
  matmul_shared_mem_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

  // DO NOT REMOVE
  CUDA_CALL(cudaDeviceSynchronize());
}

void matmul_1d_blocktiling_naive() 
{
  // DO NOT REMOVE
  CUDA_CALL(cudaSetDevice(0));

  dim3 dimGrid(N / ONE_D_BLOCKTILING_NAIVE_TILE_SIZE, M / ONE_D_BLOCKTILING_NAIVE_TILE_SIZE);
  dim3 dimBlock(ONE_D_BLOCKTILING_NAIVE_TILE_SIZE / ONE_D_BLOCKTILING_NAIVE_WPT, ONE_D_BLOCKTILING_NAIVE_TILE_SIZE);   
  matmul_1d_blocktiling_naive_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

  // DO NOT REMOVE
  CUDA_CALL(cudaDeviceSynchronize());
}

void matmul_1d_blocktiling_row() 
{
  // DO NOT REMOVE
  CUDA_CALL(cudaSetDevice(0));

  dim3 dimGrid(N / ONE_D_BLOCKTILING_ROW_TILE_SIZE, M / ONE_D_BLOCKTILING_ROW_TILE_SIZE);
  dim3 dimBlock(ONE_D_BLOCKTILING_ROW_TILE_SIZE / ONE_D_BLOCKTILING_ROW_WPT, ONE_D_BLOCKTILING_ROW_TILE_SIZE);   
  matmul_1d_blocktiling_row_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

  // DO NOT REMOVE
  CUDA_CALL(cudaDeviceSynchronize());
}

void matmul_1d_blocktiling_row_vector_type() 
{
  // DO NOT REMOVE
  CUDA_CALL(cudaSetDevice(0));

  dim3 dimGrid(N / ONE_D_BLOCKTILING_ROW_VECTOR_TYPE_TILE_SIZE, M / ONE_D_BLOCKTILING_ROW_VECTOR_TYPE_TILE_SIZE);
  dim3 dimBlock(ONE_D_BLOCKTILING_ROW_VECTOR_TYPE_TILE_SIZE / ONE_D_BLOCKTILING_ROW_VECTOR_TYPE_WPT, ONE_D_BLOCKTILING_ROW_VECTOR_TYPE_TILE_SIZE);   
  matmul_1d_blocktiling_row_vector_type_kernel<<<dimGrid, dimBlock>>>((float4*)d_A, (float4*)d_B, (float4*)d_C, M, N, K);

  // DO NOT REMOVE
  CUDA_CALL(cudaDeviceSynchronize());
}

void matmul_1d_blocktiling_column() 
{
  // DO NOT REMOVE
  CUDA_CALL(cudaSetDevice(0));

  dim3 dimGrid(N / ONE_D_BLOCKTILING_COL_TILE_SIZE, M / ONE_D_BLOCKTILING_COL_TILE_SIZE);
  dim3 dimBlock(ONE_D_BLOCKTILING_COL_TILE_SIZE, ONE_D_BLOCKTILING_COL_TILE_SIZE / ONE_D_BLOCKTILING_COL_WPT);   
  matmul_1d_blocktiling_column_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

  // DO NOT REMOVE
  CUDA_CALL(cudaDeviceSynchronize());
}

void matmul_2d_blocktiling() 
{
  // DO NOT REMOVE
  CUDA_CALL(cudaSetDevice(0));

  dim3 dimGrid(N / TWO_D_BLOCKTILING_TILE_SIZE, M / TWO_D_BLOCKTILING_TILE_SIZE);
  dim3 dimBlock(TWO_D_BLOCKTILING_TILE_SIZE / TWO_D_BLOCKTILING_CPT, TWO_D_BLOCKTILING_TILE_SIZE / TWO_D_BLOCKTILING_RPT);   
  matmul_2d_blocktiling_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

  // DO NOT REMOVE
  CUDA_CALL(cudaDeviceSynchronize());
}

void matmul_2d_blocktiling_loop_unrolling() 
{
  // DO NOT REMOVE
  CUDA_CALL(cudaSetDevice(0));

  dim3 dimGrid(N / TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE, M / TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE);
  dim3 dimBlock(TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE / TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT, TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE / TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT);   
  matmul_2d_blocktiling_loop_unrolling_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

  // DO NOT REMOVE
  CUDA_CALL(cudaDeviceSynchronize());
}

void matmul_cublas() 
{
  // DO NOT REMOVE
  CUDA_CALL(cudaSetDevice(0));
  
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));

  // DO NOT REMOVE
  CUDA_CALL(cudaDeviceSynchronize());
}