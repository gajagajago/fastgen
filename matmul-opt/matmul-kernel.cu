// Copyright (c) 2023-present, Junyeol Ryu

#include "matmul-kernel.h"
#include <assert.h>

__global__ void matmul_naive_kernel(float *d_M, float *d_N, float *d_P, int M, int N, int K)
{
  int Row = blockDim.x * blockIdx.x + threadIdx.x;
  int Col = blockDim.y * blockIdx.y + threadIdx.y;

  float acc = 0.0f;

  for (int k = 0; k < K; k++) {
    acc += d_M[Row * K + k] * d_N[k * N + Col];
  }

  d_P[Row * N + Col] = acc;
}

__global__ void matmul_global_mem_coalescing_kernel(float *d_M, float *d_N, float *d_P, int M, int N, int K)
{
  int Col = blockDim.x * blockIdx.x + threadIdx.x;
  int Row = blockDim.y * blockIdx.y + threadIdx.y;

  float acc = 0.0f;

  for (int k = 0; k < K; k++) {
    acc += d_M[Row * K + k] * d_N[k * N + Col];
  }

  d_P[Row * N + Col] = acc;
}

__global__ void matmul_shared_mem_kernel(float *d_M, float *d_N, float *d_P, int M, int N, int K) 
{
  float acc = 0.0f;  

  const int Col = blockIdx.x * blockDim.x + threadIdx.x;
  const int Row = blockIdx.y * blockDim.y + threadIdx.y;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // 1. Init shared memory data structures
  const size_t tile_width = SHARED_MEM_TILE_SIZE;
  const size_t tile_height = SHARED_MEM_TILE_SIZE;

  __shared__ float Mds[tile_height][tile_width];
  __shared__ float Nds[tile_height][tile_width];

  // 2. Init phases
  const int total_phases = K / SHARED_MEM_TILE_SIZE; 
  
  // 3. Tiling
  int t_row = ty;
  int t_col = tx;

  const int d_M_offset = Row * K;

  for (int phase = 0; phase < total_phases; phase++) {
    Mds[ty][tx] = d_M[d_M_offset + t_col];
    Nds[ty][tx] = d_N[t_row * N + Col];

    __syncthreads();    // Synchronization - Tile elements of this phase are loaded

    for (int i = 0; i < tile_width; i++) {
      float valM = Mds[ty][i]; // no bank conflict for broadcast
      float valN = Nds[i][tx];
      acc += valM * valN;      
    }

    __syncthreads();    // Synchronization - Wait all threads to finish this phase. Tile elements in shared memory will be refreshed.
  
    t_row += tile_height;
    t_col += tile_width;
  }

  d_P[Row * N + Col] = acc;
}

__global__ void matmul_1d_blocktiling_naive_kernel(float *d_M, float *d_N, float *d_P, int M, int N, int K) 
{
  float acc[ONE_D_BLOCKTILING_NAIVE_WPT] = {0.0f};

  // Thread computes d_P[Row][Col] ... d_P[Row][Col + ONE_D_BLOCKTILING_NAIVE_WPT - 1]
  const int Col = (blockIdx.x * blockDim.x + threadIdx.x) * ONE_D_BLOCKTILING_NAIVE_WPT; 
  const int Row = blockIdx.y * blockDim.y + threadIdx.y;

  assert(Row < M && Col < N);

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // 1. Init shared memory data structures
  const size_t tile_width = ONE_D_BLOCKTILING_NAIVE_TILE_SIZE;
  const size_t tile_height = ONE_D_BLOCKTILING_NAIVE_TILE_SIZE;

  __shared__ float Mds[tile_height][tile_width];
  __shared__ float Nds[tile_height][tile_width];

  // 2. Init phases
  const int total_phases = K / ONE_D_BLOCKTILING_NAIVE_TILE_SIZE; 
  
  // 3. Tiling
  int t_row = ty;
  int t_col = tx * ONE_D_BLOCKTILING_NAIVE_WPT;

  const int d_M_offset = Row * K;
  const int tile_x_offset = tx * ONE_D_BLOCKTILING_NAIVE_WPT;

  for (int phase = 0; phase < total_phases; phase++) {
    
    // Tiled load
    // Fails to coalesced load to SMEM
    for (int i = 0; i < ONE_D_BLOCKTILING_NAIVE_WPT; i++) {
      Mds[ty][tile_x_offset + i] = d_M[d_M_offset + t_col + i];
      Nds[ty][tile_x_offset + i] = d_N[t_row * N + Col + i];
    }

    __syncthreads();    // Synchronization - Tile elements of this phase are loaded

    for (int i = 0; i < tile_width; i++) {
      float valM = Mds[ty][i];

      for (int j = 0; j < ONE_D_BLOCKTILING_NAIVE_WPT; j++) {
        float valN = Nds[i][tile_x_offset + j];
        acc[j] += valM * valN;     
      }
    }

    __syncthreads();    // Synchronization - Wait all threads to finish this phase. Tile elements in shared memory will be refreshed.
  
    t_row += tile_height;
    t_col += tile_width;
  }

  for (int j = 0; j < ONE_D_BLOCKTILING_NAIVE_WPT; j++) {
    d_P[Row * N + Col + j] = acc[j];    
  }
}

/*
 * For each thread,
 * - `ONE_D_BLOCKTILING_ROW_WPT` column-wise elements are loaded to SMEM, for each `d_M` and `d_N` (to enable global mem coalescing)
 * - `ONE_D_BLOCKTILING_ROW_WPT` row-wise elements are computed
 */
__global__ void matmul_1d_blocktiling_row_kernel(float *d_M, float *d_N, float *d_P, int M, int N, int K) 
{
  float acc[ONE_D_BLOCKTILING_ROW_WPT] = {0.0f};

  const int Col = (blockIdx.x * blockDim.x + threadIdx.x) * ONE_D_BLOCKTILING_ROW_WPT;  // Starting column to compute
  const int Row = blockIdx.y * blockDim.y + threadIdx.y;

  // Boundary check
  assert(Row < M && Col + ONE_D_BLOCKTILING_ROW_WPT - 1 < N);

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // 1. Init shared memory data structures
  __shared__ float Mds[ONE_D_BLOCKTILING_ROW_TILE_SIZE][ONE_D_BLOCKTILING_ROW_TILE_SIZE];
  __shared__ float Nds[ONE_D_BLOCKTILING_ROW_TILE_SIZE][ONE_D_BLOCKTILING_ROW_TILE_SIZE];

  // 2. Init phases
  const int total_phases = K / ONE_D_BLOCKTILING_ROW_TILE_SIZE; 
  
  // 3. Tiling
  int d_M_load_starting_row = blockIdx.y * blockDim.y + (int)(ty / ONE_D_BLOCKTILING_ROW_WPT) * ONE_D_BLOCKTILING_ROW_WPT; // Will load ONE_D_BLOCKTILING_ROW_WPT consecutive rows per thread (every phase)
  int d_M_load_col = (int)(ty % ONE_D_BLOCKTILING_ROW_WPT) * blockDim.x + tx; // Will load only this col (every phase)

  int Mds_store_starting_row = (int)(ty / ONE_D_BLOCKTILING_ROW_WPT) * ONE_D_BLOCKTILING_ROW_WPT; // Will store ONE_D_BLOCKTILING_ROW_WPT consecutive rows per thread (every phase)
  int Mds_store_col = (int)(ty % ONE_D_BLOCKTILING_ROW_WPT) * blockDim.x + tx; // Will store only this col (every phase)

  int d_N_load_starting_row = (int)(ty / ONE_D_BLOCKTILING_ROW_WPT) * ONE_D_BLOCKTILING_ROW_WPT; // Will load ONE_D_BLOCKTILING_ROW_WPT consecutive rows per thread (every phase)
  int d_N_load_col = blockDim.x * blockIdx.x * ONE_D_BLOCKTILING_ROW_WPT + (int)(ty % ONE_D_BLOCKTILING_ROW_WPT) * blockDim.x + tx; // Will load only this col (every phase)

  int Nds_store_starting_row = (int)(ty / ONE_D_BLOCKTILING_ROW_WPT) * ONE_D_BLOCKTILING_ROW_WPT; // Will store ONE_D_BLOCKTILING_ROW_WPT consecutive rows per thread (every phase)
  int Nds_store_col = (int)(ty % ONE_D_BLOCKTILING_ROW_WPT) * blockDim.x + tx; // Will store only this col (every phase)

  for (int phase = 0; phase < total_phases; phase++) 
  {
    /* 
     * Load tile
     *
     * Each thread loads `ONE_D_BLOCKTILING_ROW_WPT` column-wise elements (although it will compute `ONE_D_BLOCKTILING_ROW_WPT` row-wise elements), 
     * resulting in coalesced global memory reads.
     */
    for (int i = 0; i < ONE_D_BLOCKTILING_ROW_WPT; i++) {
      int d_M_load_row = d_M_load_starting_row + i;
      int Mds_store_row = Mds_store_starting_row + i;

      Mds[Mds_store_row][Mds_store_col] = d_M[d_M_load_row * K + d_M_load_col];

      int d_N_load_row = d_N_load_starting_row + i;
      int Nds_store_row = Nds_store_starting_row + i;

      Nds[Nds_store_row][Nds_store_col] = d_N[d_N_load_row * N + d_N_load_col];
    }

    __syncthreads();    // Synchronization - Tile elements of this phase are loaded

    for (int i = 0; i < ONE_D_BLOCKTILING_ROW_TILE_SIZE; i++) {
      float valM = Mds[ty][i];

      for (int j = 0; j < ONE_D_BLOCKTILING_ROW_WPT; j++) {
        float valN = Nds[i][tx * ONE_D_BLOCKTILING_ROW_WPT + j];
        acc[j] += valM * valN;     
      }
    }

    __syncthreads();    // Synchronization - Wait all threads to finish this phase. Tile elements in shared memory will be refreshed.

    d_M_load_col += ONE_D_BLOCKTILING_ROW_TILE_SIZE;
    d_N_load_starting_row += ONE_D_BLOCKTILING_ROW_TILE_SIZE;
  }
  
  for (int j = 0; j < ONE_D_BLOCKTILING_ROW_WPT; j++) {
    d_P[Row * N + Col + j] = acc[j];    
  }
}

__global__ void matmul_1d_blocktiling_row_vector_type_kernel(float4 *d_M, float4 *d_N, float4 *d_P, int M, int N, int K) 
{

  float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};  // temporary accumulation buffer of `WPT` items

  const int Col = blockIdx.x * blockDim.x + threadIdx.x;
  const int Row = blockIdx.y * blockDim.y + threadIdx.y;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int float4_K_width = K / ONE_D_BLOCKTILING_ROW_VECTOR_TYPE_WPT;
  const int float4_N_width = N / ONE_D_BLOCKTILING_ROW_VECTOR_TYPE_WPT;

  // 1. Init shared memory data structures
  const size_t tile_width = ONE_D_BLOCKTILING_ROW_VECTOR_TYPE_TILE_SIZE / ONE_D_BLOCKTILING_ROW_VECTOR_TYPE_WPT;
  const size_t tile_height = ONE_D_BLOCKTILING_ROW_VECTOR_TYPE_TILE_SIZE;

  __shared__ float4 Mds[tile_height][tile_width];
  __shared__ float4 Nds[tile_height][tile_width];

  // 2. Init phases
  const int total_phases = K / ONE_D_BLOCKTILING_ROW_VECTOR_TYPE_TILE_SIZE; 
  
  // 3. Tiling
  int t_row = ty;
  int t_col = tx;

  const int d_M_offset = Row * float4_K_width;

  for (int phase = 0; phase < total_phases; phase++) {
    Mds[ty][tx] = d_M[d_M_offset + t_col];
    Nds[ty][tx] = d_N[t_row * float4_N_width + Col];

    __syncthreads();    // Synchronization - Tile elements of this phase are loaded

    float4 vecM, vecN;
    float valM;

    for (int i = 0; i < tile_width; i++) {
      vecM = Mds[ty][i];
      int Nds_offset = ONE_D_BLOCKTILING_ROW_VECTOR_TYPE_WPT * i;

      for (int w = 0; w < ONE_D_BLOCKTILING_ROW_VECTOR_TYPE_WPT; w++) {
        vecN = Nds[Nds_offset + w][tx];

        switch(w) {
          case 0: valM = vecM.x; break;
          case 1: valM = vecM.y; break;
          case 2: valM = vecM.z; break;
          case 3: valM = vecM.w; break;
        }

        acc.x += vecN.x * valM;
        acc.y += vecN.y * valM;
        acc.z += vecN.z * valM;
        acc.w += vecN.w * valM;
      }
    }

    __syncthreads();    // Synchronization - Wait all threads to finish this phase. Tile elements in shared memory will be refreshed.
  
    t_row += tile_height;
    t_col += tile_width;
  }

  d_P[Row * float4_N_width + Col] = acc;
}

__global__ void matmul_1d_blocktiling_column_kernel(float *d_M, float *d_N, float *d_P, int M, int N, int K) 
{
  float acc[ONE_D_BLOCKTILING_COL_WPT] = {0.0f};

  const int Col = blockIdx.x * blockDim.x + threadIdx.x; 
  const int Row = (blockIdx.y * blockDim.y + threadIdx.y) * ONE_D_BLOCKTILING_COL_WPT; // Starting row to compute

  // Boundary check
  assert(Row + ONE_D_BLOCKTILING_COL_WPT - 1 < M && Col < N);

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // 1. Init shared memory data structures
  __shared__ float Mds[ONE_D_BLOCKTILING_COL_TILE_SIZE][ONE_D_BLOCKTILING_COL_TILE_SIZE];
  __shared__ float Nds[ONE_D_BLOCKTILING_COL_TILE_SIZE][ONE_D_BLOCKTILING_COL_TILE_SIZE];

  // 2. Init phases
  const int total_phases = K / ONE_D_BLOCKTILING_COL_TILE_SIZE; 
  
  // 3. Tiling
  int d_M_load_starting_row = blockIdx.y * blockDim.y * ONE_D_BLOCKTILING_COL_WPT + ty * ONE_D_BLOCKTILING_COL_WPT; // Will load ONE_D_BLOCKTILING_COL_WPT consecutive rows per thread (every phase)
  int d_M_load_col = tx; // Will load only this col (every phase)

  int Mds_store_starting_row = ty * ONE_D_BLOCKTILING_COL_WPT; // Will store ONE_D_BLOCKTILING_COL_WPT consecutive rows per thread (every phase)
  int Mds_store_col = tx; // Will store only this col (every phase)

  int d_N_load_starting_row = ty * ONE_D_BLOCKTILING_COL_WPT; // Will load ONE_D_BLOCKTILING_COL_WPT consecutive rows per thread (every phase)
  int d_N_load_col = blockDim.x * blockIdx.x + tx; // Will load only this col (every phase)

  int Nds_store_starting_row = ty * ONE_D_BLOCKTILING_COL_WPT; // Will store ONE_D_BLOCKTILING_COL_WPT consecutive rows per thread (every phase)
  int Nds_store_col = tx; // Will store only this col (every phase)

  for (int phase = 0; phase < total_phases; phase++) 
  {
    /* 
     * Load tile
     *
     * Each thread loads `ONE_D_BLOCKTILING_COL_WPT` column-wise elements (it will compute `ONE_D_BLOCKTILING_COL_WPT` column-wise elements), 
     * resulting in coalesced global memory reads.
     */
    for (int i = 0; i < ONE_D_BLOCKTILING_COL_WPT; i++) {
      int d_M_load_row = d_M_load_starting_row + i;
      int Mds_store_row = Mds_store_starting_row + i;

      Mds[Mds_store_row][Mds_store_col] = d_M[d_M_load_row * K + d_M_load_col];

      int d_N_load_row = d_N_load_starting_row + i;
      int Nds_store_row = Nds_store_starting_row + i;

      Nds[Nds_store_row][Nds_store_col] = d_N[d_N_load_row * N + d_N_load_col];

    __syncthreads();    // Synchronization - Tile elements of this phase are loaded
    }
  
    for (int i = 0; i < ONE_D_BLOCKTILING_COL_TILE_SIZE; i++) {
      float valN = Nds[i][tx];

      for (int j = 0; j < ONE_D_BLOCKTILING_COL_WPT; j++) {
        float valM = Mds[ty * ONE_D_BLOCKTILING_COL_WPT + j][i];
        acc[j] += valM * valN;     
      }
    }

    __syncthreads();    // Synchronization - Wait all threads to finish this phase. Tile elements in shared memory will be refreshed.
  
    d_M_load_col += ONE_D_BLOCKTILING_COL_TILE_SIZE;
    d_N_load_starting_row += ONE_D_BLOCKTILING_COL_TILE_SIZE;
  }

  for (int i = 0; i < ONE_D_BLOCKTILING_COL_WPT; i++) {
    d_P[(Row + i) * N + Col] = acc[i];    
  }
}

__global__ void matmul_2d_blocktiling_kernel(float *d_M, float *d_N, float *d_P, int M, int N, int K) 
{
  float acc[TWO_D_BLOCKTILING_RPT][TWO_D_BLOCKTILING_CPT] = {0.0f};
  
  float regM[TWO_D_BLOCKTILING_RPT];
  float regN[TWO_D_BLOCKTILING_CPT];

  // Thread computes d_P[Row][Col] ... d_P[Row][Col + TWO_D_BLOCKTILING_CPT - 1]
  //                               ... 
  // d_P[Row + TWO_D_BLOCKTILING_RPT - 1][Col + TWO_D_BLOCKTILING_CPT - 1] ... d_P[Row][Col + TWO_D_BLOCKTILING_CPT - 1]
  const int Col = (blockIdx.x * blockDim.x + threadIdx.x) * TWO_D_BLOCKTILING_CPT; 
  const int Row = (blockIdx.y * blockDim.y + threadIdx.y) * TWO_D_BLOCKTILING_RPT;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // 1. Init shared memory data structures
  __shared__ float Mds[TWO_D_BLOCKTILING_TILE_SIZE][TWO_D_BLOCKTILING_TILE_SIZE];
  __shared__ float Nds[TWO_D_BLOCKTILING_TILE_SIZE][TWO_D_BLOCKTILING_TILE_SIZE];

  // 2. Init phases
  const int total_phases = K / TWO_D_BLOCKTILING_TILE_SIZE; 
  
  // 3. Tiling
  int d_M_load_starting_row = (blockIdx.y * blockDim.y * TWO_D_BLOCKTILING_CPT)
                            + (int)(ty / (TWO_D_BLOCKTILING_TILE_SIZE / blockDim.x)) * (TWO_D_BLOCKTILING_RPT * TWO_D_BLOCKTILING_CPT);
                            // Will load TWO_D_BLOCKTILING_RPT * TWO_D_BLOCKTILING_CPT consecutive rows per thread (every phase)
  int d_M_load_col = (int)(ty % TWO_D_BLOCKTILING_RPT) * blockDim.x + tx; // Will load only this col (every phase)

  int Mds_store_starting_row = (int)(ty / (TWO_D_BLOCKTILING_TILE_SIZE / blockDim.x)) * (TWO_D_BLOCKTILING_RPT * TWO_D_BLOCKTILING_CPT); // Will store TWO_D_BLOCKTILING_RPT * TWO_D_BLOCKTILING_CPT consecutive rows per thread (every phase)
  int Mds_store_col = (int)(ty % TWO_D_BLOCKTILING_RPT) * blockDim.x + tx; // Will store only this col (every phase)

  int d_N_load_starting_row = (int)(ty / (TWO_D_BLOCKTILING_TILE_SIZE / blockDim.x)) * (TWO_D_BLOCKTILING_RPT * TWO_D_BLOCKTILING_CPT); 
                            // Will load TWO_D_BLOCKTILING_RPT * TWO_D_BLOCKTILING_CPT consecutive rows per thread (every phase)
  int d_N_load_col = blockDim.x * blockIdx.x * TWO_D_BLOCKTILING_CPT + (int)(ty % TWO_D_BLOCKTILING_RPT) * blockDim.x + tx; // Will load only this col (every phase)

  int Nds_store_starting_row = (int)(ty / (TWO_D_BLOCKTILING_TILE_SIZE / blockDim.x)) * (TWO_D_BLOCKTILING_RPT * TWO_D_BLOCKTILING_CPT); // Will store TWO_D_BLOCKTILING_RPT * TWO_D_BLOCKTILING_CPT consecutive rows per thread (every phase)
  int Nds_store_col = (int)(ty % TWO_D_BLOCKTILING_RPT) * blockDim.x + tx; // Will store only this col (every phase)

  for (int phase = 0; phase < total_phases; phase++) 
  {  
    /* 
     * Load tile
     *
     * Each thread loads `TWO_D_BLOCKTILING_RPT * TWO_D_BLOCKTILING_CPT` column-wise elements 
     * (it will compute `TWO_D_BLOCKTILING_RPT * TWO_D_BLOCKTILING_CPT` 2D elements), 
     * resulting in coalesced global memory reads.
     */    
    for (int i = 0; i < TWO_D_BLOCKTILING_RPT; i++) {
      for (int j = 0; j < TWO_D_BLOCKTILING_CPT; j++) {

        int d_M_load_row = d_M_load_starting_row + (i * TWO_D_BLOCKTILING_CPT) + j;
        int Mds_store_row = Mds_store_starting_row + (i * TWO_D_BLOCKTILING_CPT) + j;

        Mds[Mds_store_row][Mds_store_col] = d_M[d_M_load_row * K + d_M_load_col];

        int d_N_load_row = d_N_load_starting_row + (i * TWO_D_BLOCKTILING_CPT) + j;
        int Nds_store_row = Nds_store_starting_row + (i * TWO_D_BLOCKTILING_CPT) + j;

        Nds[Nds_store_row][Nds_store_col] = d_N[d_N_load_row * N + d_N_load_col];
      }
    }

    __syncthreads();    // Synchronization - Tile elements of this phase are loaded

    for (int dotIdx = 0; dotIdx < TWO_D_BLOCKTILING_TILE_SIZE; dotIdx++) {

      for (int i = 0; i < TWO_D_BLOCKTILING_RPT; i++) {
        regM[i] = Mds[ty  * TWO_D_BLOCKTILING_RPT + i][dotIdx];
      }
      for (int j = 0; j < TWO_D_BLOCKTILING_CPT; j++) {
        regN[j] = Nds[dotIdx][tx * TWO_D_BLOCKTILING_CPT + j];
      }

      for (int regM_idx = 0; regM_idx < TWO_D_BLOCKTILING_RPT; regM_idx++) {
        for (int regN_idx = 0; regN_idx < TWO_D_BLOCKTILING_CPT; regN_idx++) {
          acc[regM_idx][regN_idx] += regM[regM_idx] * regN[regN_idx]; // Expected problem: Too slow GMEM access?
        }
      }
    }

    __syncthreads();    // Synchronization - Wait all threads to finish this phase. Tile elements in shared memory will be refreshed.

    d_M_load_col += TWO_D_BLOCKTILING_TILE_SIZE;
    d_N_load_starting_row += TWO_D_BLOCKTILING_TILE_SIZE;
  }

  for (int i = 0; i < TWO_D_BLOCKTILING_RPT; i++) {
    for (int j = 0; j < TWO_D_BLOCKTILING_CPT; j++) {
      d_P[(Row+i) * N + Col+j] = acc[i][j];   
    } 
  }
}

__global__ void matmul_2d_blocktiling_loop_unrolling_kernel(float *d_M, float *d_N, float *d_P, int M, int N, int K) 
{
  float acc[TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT][TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT] = {0.0f};
  
  float regM[TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT];
  float regN[TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT];

  const int Col = (blockIdx.x * blockDim.x + threadIdx.x) * TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT; 
  const int Row = (blockIdx.y * blockDim.y + threadIdx.y) * TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // 1. Init shared memory data structures
  __shared__ float Mds[TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE][TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE];
  __shared__ float Nds[TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE][TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE];

  // 2. Init phases
  const int total_phases = K / TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE; 
  
  // 3. Tiling
  int d_M_load_starting_row = (blockIdx.y * blockDim.y * TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT)
                            + (int)(ty / (TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE / blockDim.x)) * (TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT * TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT);
                            // Will load TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT * TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT consecutive rows per thread (every phase)
  int d_M_load_col = (int)(ty % TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT) * blockDim.x + tx; // Will load only this col (every phase)

  int Mds_store_starting_row = (int)(ty / (TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE / blockDim.x)) * (TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT * TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT); // Will store TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT * TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT consecutive rows per thread (every phase)
  int Mds_store_col = (int)(ty % TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT) * blockDim.x + tx; // Will store only this col (every phase)

  int d_N_load_starting_row = (int)(ty / (TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE / blockDim.x)) * (TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT * TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT); 
                            // Will load TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT * TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT consecutive rows per thread (every phase)
  int d_N_load_col = blockDim.x * blockIdx.x * TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT + (int)(ty % TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT) * blockDim.x + tx; // Will load only this col (every phase)

  int Nds_store_starting_row = (int)(ty / (TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE / blockDim.x)) * (TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT * TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT); // Will store TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT * TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT consecutive rows per thread (every phase)
  int Nds_store_col = (int)(ty % TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT) * blockDim.x + tx; // Will store only this col (every phase)

  #pragma unroll
  for (int phase = 0; phase < total_phases; phase++) 
  {  
    /* 
     * Load tile
     *
     * Each thread loads `TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT * TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT` column-wise elements 
     * (it will compute `TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT * TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT` 2D elements), 
     * resulting in coalesced global memory reads.
     */    
    #pragma unroll
    for (int i = 0; i < TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT; i++) {
      #pragma unroll
      for (int j = 0; j < TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT; j++) {

        int d_M_load_row = d_M_load_starting_row + (i * TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT) + j;
        int Mds_store_row = Mds_store_starting_row + (i * TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT) + j;

        Mds[Mds_store_row][Mds_store_col] = d_M[d_M_load_row * K + d_M_load_col];

        int d_N_load_row = d_N_load_starting_row + (i * TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT) + j;
        int Nds_store_row = Nds_store_starting_row + (i * TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT) + j;

        Nds[Nds_store_row][Nds_store_col] = d_N[d_N_load_row * N + d_N_load_col];
      }
    }

    __syncthreads();    // Synchronization - Tile elements of this phase are loaded

    #pragma unroll
    for (int dotIdx = 0; dotIdx < TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE; dotIdx++) {
      #pragma unroll
      for (int i = 0; i < TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT; i++) {
        regM[i] = Mds[ty  * TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT + i][dotIdx];
      }
      #pragma unroll
      for (int j = 0; j < TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT; j++) {
        regN[j] = Nds[dotIdx][tx * TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT + j];
      }

      #pragma unroll
      for (int regM_idx = 0; regM_idx < TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT; regM_idx++) {
        #pragma unroll
        for (int regN_idx = 0; regN_idx < TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT; regN_idx++) {
          acc[regM_idx][regN_idx] += regM[regM_idx] * regN[regN_idx]; // Expected problem: Too slow GMEM access?
        }
      }
    }

    __syncthreads();    // Synchronization - Wait all threads to finish this phase. Tile elements in shared memory will be refreshed.

    d_M_load_col += TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE;
    d_N_load_starting_row += TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE;
  }

  #pragma unroll
  for (int i = 0; i < TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT; i++) {
    #pragma unroll
    for (int j = 0; j < TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT; j++) {
      d_P[(Row+i) * N + Col+j] = acc[i][j];   
    } 
  }
}