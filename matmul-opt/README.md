# Optimizing a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog

## Setup
1 V100 32GB GPU 
- Peak FP32 GFLOPS: 15700
- L1 cache + SMEM size: 128KB/SM
- Configured SMEM size: 96KB/SM
- Peak GMEM bandwidth: 900GB/s
- Peak SMEM bandwidth: 12080GB/s (reported)

## Performance comparison
Kernel | GFLOPS | Performance relative to cuBLAS
---------- | ---------- | ----------
1: Naive | 214.5 | 1.5%
2: Global mem coalescing | 2084.1 | 14.8%
3: Shared mem | 3809.3 | 27.2%
4: 1D block tiling (naive) | 4950.2 | 35.3%
5: 1D block tiling (row) | 5005.3 | 35.7%
6: 1D block tiling (row, vector type) | 5093.1 | 36.3%
7: 1D block tiling (column) | 5754.0 | 41.1%
8: 2D block tiling | 10440.7 | 74.5%
9: 2D block tiling (loop unrolling) | 11344.2 | 80.9%
0: cuBLAS | 14015.0 | 100.0%

# Future optimizations
- Warp tiling
- Tensor core