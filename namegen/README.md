# Namegen

## How to run
```
./run.sh model.bin <output path> <num names> <seed>
```

## Applied optimizations
1. Tiling, Global memory coalescing [5]
2. CUDA Vector types
3. Loop unrolling
4. Pinned memory [1]
- cudaMemcpy from host to device takes 3 stages: 1) Allocate a temporary page-locked("pinned") host array 2) Copy the host data to the pinned array 3) Transfer the data from the pinned array to the device memory.
- Directly allocating the host arrays in pinned memory using `cudaMallocHost()` or `cudaHostAlloc()`, and deallocating with `cudaFreeHost()` reduces overhead in 1) and 2). 
5. CUDA Streams
- The default stream("null stream") is the synchronizing stream: no operation in the default stream will begin until all previously issued operations in any stream on the device have completed, and an operation in the default stream must complete before any other operation (in any stream on the device) will begin. 

## Possible optimizations
1. Double buffering
- The kernel execution and the data transfer to be overlapped must both occur in different, non-default streams.
- The host memory involved in the data transfer must be pinned memory.
- Memcpy and Memset API are synchronous by default [3].
- To issue a data transfer to a non-default stream we use the `cudaMemcpyAsync()` function which takes a stream identifier as a fifth argument. To issue a kernel to a non-default stream we specify the stream identifier as a fourth execution configuration parameter.

## Dropped optimiations
1. Kernel fusion

## Evaluation

### Environment
We run our evaluation using a server of 4 machines, each equipped with 4 NVIDIA V100 32GB GPUs connected over PCIe. Each machine has 1 Mellanox MT28908 family ConnectX-6 NIC and is interconnected with NVIDIA QUANTUM HDR Switch QM8700, providing an 200Gb/s of interconnect bandwidth between the machines.

## Performance
580201.503 names/sec

## References
1. [How to Optimize Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
2. [How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
3. [API Synchronization Behavior](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html)
4. [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
5. [Matrix Multiplication CUDA](https://ecatue.gitlab.io/gpu2018/pages/Cookbook/matrix_multiplication_cuda.html)