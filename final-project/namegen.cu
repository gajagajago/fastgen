// Copyright (c) 2023-present, Junyeol Ryu

#include "namegen.h"
#include "util.h"
#include "matmul-kernel.h"

#include <cassert>
#include <math.h>
#include <vector>
#include <omp.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <iterator>

////////////////////////////// NVTX ////////////////////////////////////////

#include <nvToolsExt.h>
unsigned int colors[16] = {
  0xFF00FF00, 0xFF008000, 0xFF00FFFF, 0xFF008080, 0xFF0000FF, 0xFF000080,
  0xFFFF00FF, 0xFF800080, 0xFFFFFFFF, 0xFFC0C0C0, 0xFF808080, 0xFF000000,
  0xFFFF0000, 0xFF800000, 0xFFFFFF00, 0xFF808000,
};

char nvtx_msg[16][64];
nvtxRangeId_t nvtx_range_ids[16];

void startNVTXEvent(const char *message) {
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message;

  int idx = -1;
  for (int i = 0; i < 16; i++) {
    if (strcmp(nvtx_msg[i], message) == 0) {
      eventAttrib.color = colors[i];
      idx = i;
      break;
    }
  }

  if (idx == -1) {
    for (int i = 0; i < 16; i++) {
      if (nvtx_msg[i][0] == 0) {
        strcpy(nvtx_msg[i], message);
        eventAttrib.color = colors[i];
        idx = i;
        break;
      }
    }
  }
  nvtx_range_ids[idx] = nvtxRangeStartEx(&eventAttrib);
}

void stopNVTXEvent(const char *message) {
  for (int i = 0; i < 16; i++) {
    if (strcmp(nvtx_msg[i], message) == 0) {
      nvtxRangeEnd(nvtx_range_ids[i]);
      return;
    }
  }
}


////////////////////////////// NVTX END ////////////////////////////////////////

/* 
 * Logging rules
 *
 * LOG1: General progress/step info
 * LOG2: Tensor buffer/device buffer print from `namegen`
 * LOG3: Misc (e.g., kernel launch configs), Tensor buffer/device buffer print from else than `namegen` 
 * LOG_TIME: Step latency
 */
// #define DEBUG1     // Comment this line to disable LOG1 & LOG2 & LOG3
// #define DEBUG2     // Comment this line to disable LOG2 & LOG3
// #define DEBUG3     // Comment this line to disable LOG3
// #define DEBUG_TIME // Comment this line to disable LOG_TIME

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

#if defined DEBUG1 && defined DEBUG2 && defined DEBUG3
#define LOG3 1
#else
#define LOG3 0
#endif 

#if defined DEBUG_TIME
#define LOG_TIME 1
#else
#define LOG_TIME 0
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

#define OMP_MAX_THREADS 64
#define MAX_NUM_GPU 4

static size_t CUDA_MAX_SHARED_MEM_SIZE;
static size_t CUDA_MAX_THREADS_PER_BLOCK;
static int BSZ; 

/*
 * Matmul kernel configuration
 * 
 * TILE_SIZE: Size of a tile's side. Each thread block is assigned TILE_SIZE * TILE_SIZE elements.
 * WPT: Number of output elements assigned to a thread. `TILE_SIZE` must be divisible by `WPT`.
 */
#define TILE_SIZE 64
#define WPT 4

/* 
* MPI main data structures
*
* mpi_rank: Rank of each MPI process
* mpi_size: Size of the MPI communicator
*/
extern int mpi_rank, mpi_size; // Defined in main.cpp
#define IS_MPI_ROOT mpi_rank == 0

static int num_devices = 0;
cudaStream_t* device_stream;

__global__ void embedding_kernel(float* in, float4 *lookup, float4 *out, int M, int K);
__global__ void sampling_kernel(float *in, float *rfloats, float *out, int K, int N, int l);
__global__ void transpose_kernel(float* in, float* out, int m, int n);
__global__ void matvecadd_kernel(float4 *d_M, float4 *d_N, float4 *d_P, int M, int K);
__global__ void matadd_kernel(float4 *d_M, float4 *d_N, float4 *d_P, int M, int K);
__global__ void sigmoid_kernel(float4 *d_M, float4 *d_P, int M, int K);
__global__ void mathadamardproduct_kernel(float4 *d_M, float4 *d_N, float4 *d_P, int M, int K);
__global__ void tanh_kernel(float4 *d_M, float4 *d_P, int M, int K);
__global__ void oneminus_kernel(float4 *d_M, float4 *d_P, int M, int K);
__global__ void sum_reduce_kernel(float* in, float* out, int M, int K);
__global__ void exp_kernel(float4 *d_M, float4 *d_P, int M, int K);
__global__ void matvecdivide_kernel(float4 *d_M, float *d_N, float4 *d_P, int M, int K);
__global__ void memset_kernel(float4 *d_M, int M, int K, float c);

/* Utility function to log elapsed time */
void print_elapsed_time(double t_start, double t_end, const char* task)
{ 
  if (LOG_TIME)
    PRINTF_WITH_RANK("%s: %.6f seconds", task, t_end - t_start);
}

/*
 * Tensor
 */
struct Tensor {

  float *buf = nullptr;
  float **device_buf = nullptr;

  size_t ndim = 0; // buf_ndim == device_buf_ndim

  size_t buf_shape[2];
  size_t device_buf_shape[2];

  /* Alloc memory */
  Tensor(std::vector<int> buf_shape_) {
    ndim = buf_shape_.size();
    assert(ndim > 0 && ndim <= 2);

    for (size_t i = 0; i < ndim; i++) {
      buf_shape[i] = buf_shape_[i];
    }

    buf = (float*)malloc(buf_num_elem() * sizeof(float));
    device_buf = (float **)malloc(num_devices * sizeof(float*));
  }

  /* Alloc memory and copy */
  Tensor(std::vector<int> buf_shape_, float *buf_) {
    ndim = buf_shape_.size();
    assert(ndim > 0 && ndim <= 2);

    for (size_t i = 0; i < ndim; i++) {
      buf_shape[i] = buf_shape_[i];
    }

    buf = (float*)malloc(buf_num_elem() * sizeof(float));
    memcpy(buf, buf_, buf_num_elem() * sizeof(float));
    device_buf = (float **)malloc(num_devices * sizeof(float*));
  }

  ~Tensor() {
    if (buf != nullptr) 
      free(buf);
    if (device_buf != nullptr)
      free(device_buf);
  }

  void alloc_device_buf(int device, std::vector<int> device_buf_shape_) {
    CUDA_CALL(cudaSetDevice(device));

    ndim = device_buf_shape_.size();
    for (size_t i = 0; i < ndim; i++) {
      device_buf_shape[i] = device_buf_shape_[i];
    }

    CUDA_CALL(cudaMalloc(&device_buf[device], device_buf_num_elem() * sizeof(float)));
  }

  void set_zero() {
    size_t n = buf_num_elem();
    memset(buf, 0.0, buf_num_elem() * sizeof(float));
  }

  void device_buf_set(int device, float c) {
    CUDA_CALL(cudaSetDevice(device));

    if (ndim == 1) {
      int num_threads = MIN(device_buf_shape[0]/WPT, CUDA_MAX_THREADS_PER_BLOCK) ;

      dim3 dimGrid((device_buf_shape[0]/WPT) / num_threads);
      dim3 dimBlock(num_threads);

      memset_kernel<<<dimGrid, dimBlock, 0, device_stream[device]>>>((float4*)device_buf[device], 1, device_buf_shape[0], c);
    } else {
      int blockDim_x = MIN(device_buf_shape[1]/WPT, CUDA_MAX_THREADS_PER_BLOCK);
      int blockDim_y = CUDA_MAX_THREADS_PER_BLOCK / blockDim_x;
      
      dim3 dimGrid((device_buf_shape[1]/WPT)/blockDim_x, device_buf_shape[0]/blockDim_y);
      dim3 dimBlock(blockDim_x, blockDim_y);

      memset_kernel<<<dimGrid, dimBlock, 0, device_stream[device]>>>((float4*)device_buf[device], device_buf_shape[0], device_buf_shape[1], c);
    }
  }

  size_t buf_num_elem() {
    size_t sz = 1;
    for (size_t i = 0; i < ndim; i++)
      sz *= buf_shape[i];
    return sz;
  }

  size_t device_buf_num_elem() {
    size_t sz = 1;
    for (size_t i = 0; i < ndim; i++)
      sz *= device_buf_shape[i];
    return sz;
  }

  // copy buffer to device buffer
  void to(int device) {
    assert(num_devices > 0);
    assert(device > -1 && device < num_devices);

    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaMemcpyAsync(device_buf[device], (void*)buf, buf_num_elem() * sizeof(float), cudaMemcpyHostToDevice, device_stream[device]));
  }

  // copy part of buffer to device buffer
  void to(int device, int from) {
    assert(num_devices > 0);
    assert(device > -1 && device < num_devices);

    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaMemcpyAsync(device_buf[device], (void*)&buf[from], device_buf_num_elem() * sizeof(float), cudaMemcpyHostToDevice, device_stream[device]));
  }

  // copy device buffer to buffer
  void from(int device, int to) {
    assert(device_buf[device] != nullptr);

    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaMemcpyAsync((void*)&buf[to], device_buf[device], device_buf_num_elem() * sizeof(float), cudaMemcpyDeviceToHost, device_stream[device]));
  }

  // copy device buffer to other tensor's device buffer
  // Two dimensional memcpy of lower-right elements starting from elem[row,col]
  void device_buf_copy(int device, Tensor* tensor, int row, int col) {
    assert(device_buf_shape[1] - col >= tensor->device_buf_shape[1]);
    assert(device_buf_shape[0] - row >= tensor->device_buf_shape[0]);

    CUDA_CALL(cudaSetDevice(device));
    
    size_t dpitch = tensor->device_buf_shape[1] * sizeof(float);
    size_t spitch = device_buf_shape[1] * sizeof(float);
    size_t width = (MIN(device_buf_shape[1] - col, tensor->device_buf_shape[1])) * sizeof(float);
    size_t height = MIN(device_buf_shape[0] - row, tensor->device_buf_shape[0]);

    CUDA_CALL(cudaMemcpy2DAsync(tensor->device_buf[device], dpitch, 
                          &device_buf[device][row*device_buf_shape[1]+col], spitch, 
                          width, height, cudaMemcpyDeviceToDevice, device_stream[device]));
  }

  // Transpose 2D buffer
  void transpose() {
    assert(ndim == 2);

    size_t blockDim_x = TILE_SIZE;
    size_t blockDim_y = MIN((CUDA_MAX_SHARED_MEM_SIZE / sizeof(float)) / blockDim_x, CUDA_MAX_THREADS_PER_BLOCK / blockDim_x);

    float* d_in;
    float* d_out;
    
    cudaMalloc( (void**) &d_in, sizeof(float) * buf_num_elem());
    cudaMalloc( (void**) &d_out, sizeof(float) * buf_num_elem());

    // copy host memory to device
    cudaMemcpy( d_in, buf, sizeof(float) * buf_num_elem(), cudaMemcpyHostToDevice);

    // setup execution parameters
    dim3 dimGrid(buf_shape[1] / blockDim_x, buf_shape[0] / blockDim_y);
    dim3 dimBlock(blockDim_x, blockDim_y);

    int maxSharedMemBytes = blockDim_x*blockDim_y*sizeof(float);
    CUDA_CALL(cudaFuncSetAttribute(transpose_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedMemBytes));
    transpose_kernel<<<dimGrid, dimBlock, maxSharedMemBytes>>>(d_in, d_out, buf_shape[0], buf_shape[1]);

    cudaMemcpy( buf, d_out, sizeof(float)*buf_num_elem(), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    // Reverse shape
    int aux[ndim];
 
    for (int i = 0; i < (int)ndim; i++) {
      aux[ndim - 1 - i] = buf_shape[i];
    }
 
    for (int i = 0; i < (int)ndim; i++) {
      buf_shape[i] = aux[i];
    }
  }

  void print_tensor_info(const char* name) {
    int i;
    printf("\n\n");
    printf("%s\n", name);
    printf("  Ndim: %d\n", (int)ndim);
    printf("  Shape: "); 
    for (i = 0; i < (int)ndim; i++) {
      printf("%d ", (int)buf_shape[i]);
    }
    printf("\n");
    for (int nd = 0; nd < (int)ndim; nd++) {
      printf("\t");
      for (i = 0; i < 3 && i < (int)buf_shape[ndim-1]; i++) {
        printf("%.3f ", buf[nd * buf_shape[ndim-1] + i]);
      }
      printf("\n");
    }
  }

  void print_device_buf_info(int device, const char* name) {

    CUDA_CALL(cudaSetDevice(device));
    
    float* h_array = (float*)malloc(sizeof(float) * device_buf_num_elem());
    cudaMemcpy(h_array, device_buf[device], device_buf_num_elem() * sizeof(float), cudaMemcpyDeviceToHost);

    int i;
    printf("\n\n");
    printf("%s[%d]\n", name, device);
    printf("  Ndim: %d\n", (int)ndim);
    printf("  Shape: "); 
    for (i = 0; i < (int)ndim; i++) {
      printf("%d ", (int)device_buf_shape[i]);
    }
    printf("\n");
    for (int nd = 0; nd < (int)ndim; nd++) {
      printf("\t");
      for (i = 0; i < 3 && i < (int)device_buf_shape[ndim-1]; i++) {
        printf("%.3f ", h_array[nd * device_buf_shape[ndim-1] + i]);
      }
      printf("\n");
    }

    free(h_array);
  }
};

/* Operations */

/*
 * Set device buffer to uniform value `c`
 */
void device_buf_set(Tensor *input, float c) 
{
  omp_set_num_threads(num_devices);

  #pragma omp parallel for
  for (int i=0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));

    input->device_buf_set(i, c);

    CUDA_CALL(cudaStreamSynchronize(device_stream[i]));
  }
}

/*
 * Copy batch_random_floats to device buf
 *
 * IN batch - batch index
 */
void copy_batch_rfloats(Tensor* batch_rfloats, int batch)
{
  omp_set_num_threads(num_devices);

  #pragma omp parallel for
  for (int i=0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));

    int offset = ( batch * (BSZ) + BSZ / (mpi_size * num_devices) * (mpi_rank * num_devices + i) ) * MAX_LEN;
    batch_rfloats->to(i, offset);

    CUDA_CALL(cudaStreamSynchronize(device_stream[i]));
  }
}

/*
 * Embedding lookup
 * input: [batch]
 * lookup: [NUM_CHAR, EMBEDDING_DIM]
 * output: [batch, EMBEDDING_DIM]
 */
void embedding(Tensor *input, Tensor *lookup, Tensor *output) 
{
  omp_set_num_threads(num_devices);

  #pragma omp parallel for
  for (int i=0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));

    assert(input->device_buf_shape[0] == output->device_buf_shape[0]);
    assert(lookup->device_buf_shape[1] == output->device_buf_shape[1]);

    int blockDim_x = MIN(output->device_buf_shape[1]/WPT, CUDA_MAX_THREADS_PER_BLOCK); // <-- start  
    int blockDim_y = CUDA_MAX_THREADS_PER_BLOCK / blockDim_x;

    dim3 dimGrid((output->device_buf_shape[1]/WPT) / blockDim_x, output->device_buf_shape[0]/blockDim_y);
    dim3 dimBlock(blockDim_x, blockDim_y);

    embedding_kernel<<<dimGrid, dimBlock, 0, device_stream[i]>>>(input->device_buf[i], (float4*)lookup->device_buf[i], 
                                          (float4*)output->device_buf[i], output->device_buf_shape[0], output->device_buf_shape[1]);

    CUDA_CALL(cudaStreamSynchronize(device_stream[i]));
  }
}

/*
 * Elementwise matrix addition
 * input1: [*]
 * input2: [*] (same shape as input1)
 * output: [*] (same shape as input1)
 */
void matadd(Tensor *input1, Tensor *input2, Tensor *output) 
{
  omp_set_num_threads(num_devices);

  #pragma omp parallel for
  for (int i=0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));

    assert(input1->device_buf_shape[0] == input2->device_buf_shape[0] && 
          input2->device_buf_shape[0] == output->device_buf_shape[0]);
    assert(input1->device_buf_shape[1] == input2->device_buf_shape[1] &&
          input2->device_buf_shape[1] == output->device_buf_shape[1]);

    int M = input1->device_buf_shape[0];
    int K = input1->device_buf_shape[1];

    dim3 dimGrid(K / TILE_SIZE, M / TILE_SIZE);
    dim3 dimBlock(TILE_SIZE / WPT, TILE_SIZE);

    matadd_kernel<<<dimGrid, dimBlock, 0, device_stream[i]>>>((float4*)input1->device_buf[i], (float4*)input2->device_buf[i], 
                                        (float4*)output->device_buf[i], M, K);

    CUDA_CALL(cudaStreamSynchronize(device_stream[i]));
  }
}

/*
 * Add vector to every row of matrix
 * input1: [M x K]
 * input2: [K] (same shape as input1)
 * output: [M x K] (same shape as input1)
 */
void matvecadd(Tensor *input1, Tensor *input2, Tensor *output) 
{
  omp_set_num_threads(num_devices);

  #pragma omp parallel for
  for (int i=0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));

    assert(input1->device_buf_shape[1] == input2->device_buf_shape[0]);

    int M = input1->device_buf_shape[0];
    int K = input2->device_buf_shape[0];

    dim3 dimGrid(K / TILE_SIZE, M / TILE_SIZE);
    dim3 dimBlock(TILE_SIZE / WPT, TILE_SIZE);

    matvecadd_kernel<<<dimGrid, dimBlock, 0, device_stream[i]>>>((float4*)input1->device_buf[i], (float4*)input2->device_buf[i], 
                                          (float4*)output->device_buf[i], M, K);

    CUDA_CALL(cudaStreamSynchronize(device_stream[i]));
  }
}

/*
 * Elementwise (1-x)
 * input: [*]
 * output: [*] (same shape as input)
 */
void oneminus(Tensor *input, Tensor *output) 
{
  omp_set_num_threads(num_devices);
  
  #pragma omp parallel for
  for (int i=0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));

    assert(input->device_buf_shape[0] == output->device_buf_shape[0]);
    assert(input->device_buf_shape[1] == output->device_buf_shape[1]);

    int M = input->device_buf_shape[0];
    int K = input->device_buf_shape[1];

    dim3 dimGrid(K / TILE_SIZE, M / TILE_SIZE);
    dim3 dimBlock(TILE_SIZE / WPT, TILE_SIZE);

    oneminus_kernel<<<dimGrid, dimBlock, 0, device_stream[i]>>>((float4*)input->device_buf[i], (float4*)output->device_buf[i], 
                                        M, K);

    CUDA_CALL(cudaStreamSynchronize(device_stream[i]));
  }
}

/*
 * Elementwise matix Hadamard product
 * input1: [*]
 * input2: [*] (same shape as input1)
 * output: [*] (same shape as input1)
 */
void mathadamardproduct(Tensor *input1, Tensor *input2, Tensor *output) 
{
  omp_set_num_threads(num_devices);

  #pragma omp parallel for
  for (int i=0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));

    assert(input1->device_buf_shape[0] == input2->device_buf_shape[0] && 
          input2->device_buf_shape[0] == output->device_buf_shape[0]);
    assert(input1->device_buf_shape[1] == input2->device_buf_shape[1] &&
          input2->device_buf_shape[1] == output->device_buf_shape[1]);

    int M = input1->device_buf_shape[0];
    int K = input1->device_buf_shape[1];

    dim3 dimGrid(K / TILE_SIZE, M / TILE_SIZE);
    dim3 dimBlock(TILE_SIZE / WPT, TILE_SIZE);

    mathadamardproduct_kernel<<<dimGrid, dimBlock, 0, device_stream[i]>>>((float4*)input1->device_buf[i], (float4*)input2->device_buf[i], 
                                        (float4*)output->device_buf[i], M, K);

    CUDA_CALL(cudaStreamSynchronize(device_stream[i]));
  }
}

/*
 * Elementwise tanh(x)
 * input: [*]
 * output: [*] (same shape as input)
 */
void tanh(Tensor *input, Tensor *output) 
{
  omp_set_num_threads(num_devices);
  
  #pragma omp parallel for
  for (int i=0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));

    assert(input->device_buf_shape[0] == output->device_buf_shape[0]);
    assert(input->device_buf_shape[1] == output->device_buf_shape[1]);

    int M = input->device_buf_shape[0];
    int K = input->device_buf_shape[1];

    dim3 dimGrid(K / TILE_SIZE, M / TILE_SIZE);
    dim3 dimBlock(TILE_SIZE / WPT, TILE_SIZE);

    tanh_kernel<<<dimGrid, dimBlock, 0, device_stream[i]>>>((float4*)input->device_buf[i], (float4*)output->device_buf[i], 
                                        M, K);

    CUDA_CALL(cudaStreamSynchronize(device_stream[i]));
  }
}

/*
 * Elementwise sigmoid 1 / (1 + exp(-x))
 * input: [*]
 * output: [*] (same shape as input)
 */
void sigmoid(Tensor *input, Tensor *output) 
{
  omp_set_num_threads(num_devices);
  
  #pragma omp parallel for
  for (int i=0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));

    assert(input->device_buf_shape[0] == output->device_buf_shape[0]);
    assert(input->device_buf_shape[1] == output->device_buf_shape[1]);

    int M = input->device_buf_shape[0];
    int K = input->device_buf_shape[1];

    dim3 dimGrid(K / TILE_SIZE, M / TILE_SIZE);
    dim3 dimBlock(TILE_SIZE / WPT, TILE_SIZE);

    sigmoid_kernel<<<dimGrid, dimBlock, 0, device_stream[i]>>>((float4*)input->device_buf[i], (float4*)output->device_buf[i], 
                                        M, K);

    CUDA_CALL(cudaStreamSynchronize(device_stream[i]));
  }
}

/*
 * Transfer of input tensor device buffer to output tensor device buffer
 * The copy starts from (row,col) index element and covering its lower-right elements
 * input: [*]
 * output: [!] 
 */
void device_buf_copy(Tensor *input, Tensor *output, int row, int col) 
{
  omp_set_num_threads(num_devices);
  
  #pragma omp parallel for
  for (int i=0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));

    input->device_buf_copy(i, output, row, col);  

    CUDA_CALL(cudaStreamSynchronize(device_stream[i]));
  }
}

/*
 * Softmax
 * input: [batch, *]
 * output: [batch, *], 
 */
void softmax(Tensor *input, Tensor *batch_reduced, Tensor *output) 
{
  omp_set_num_threads(num_devices);

  int M = input->device_buf_shape[0];
  int K = input->device_buf_shape[1];

  #pragma omp parallel for
  for (int i=0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));

    assert(input->device_buf_shape[0] == output->device_buf_shape[0]);
    assert(input->device_buf_shape[1] == output->device_buf_shape[1]);

    // exp(input)
    dim3 dimGrid(K / TILE_SIZE, M / TILE_SIZE);
    dim3 dimBlock(TILE_SIZE / WPT, TILE_SIZE);

    exp_kernel<<<dimGrid, dimBlock, 0, device_stream[i]>>>((float4*)input->device_buf[i], (float4*)input->device_buf[i], 
                                        M, K);
    
    // exp sum reduce input to reduction vector
    int blockDim_x = NUM_CHAR;
    int blockDim_y = CUDA_MAX_THREADS_PER_BLOCK / blockDim_x;

    dim3 dimGrid_reduce(K / blockDim_x, M / blockDim_y);
    dim3 dimBlock_reduce(blockDim_x, blockDim_y);
    size_t shared_memory_size = blockDim_x * blockDim_y * sizeof(float);

    sum_reduce_kernel<<<dimGrid_reduce, dimBlock_reduce, shared_memory_size, device_stream[i]>>>(input->device_buf[i], batch_reduced->device_buf[i], M, K);

    // launch matvecdivide kernel with batch_reduced
    matvecdivide_kernel<<<dimGrid, dimBlock, 0, device_stream[i]>>>((float4*)input->device_buf[i], batch_reduced->device_buf[i], (float4*)output->device_buf[i], 
                                        M, K);

    CUDA_CALL(cudaStreamSynchronize(device_stream[i]));
  }
}

/*
 * Sampling
 * 
 * input: [batch, NUM_CHAR], probability distribution of the characters
 * rfloats: [batch, MAX_LEN]
 * output: [batch]
 * l: current letter index
 */
void batch_sampling(Tensor *input, Tensor *rfloats, Tensor *output, int l) 
{
  omp_set_num_threads(num_devices);
  
  #pragma omp parallel for
  for (int i=0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));

    assert(input->device_buf_shape[0] == rfloats->device_buf_shape[0]);
    assert(input->device_buf_shape[0] == output->device_buf_shape[0]);

    //set to 255 for default
    output->device_buf_set(i, NUM_CHAR-1);

    int num_threads = MIN(output->device_buf_shape[0], CUDA_MAX_THREADS_PER_BLOCK) ;
    dim3 dimGrid(output->device_buf_shape[0] / num_threads);
    dim3 dimBlock(num_threads);

    sampling_kernel<<<dimGrid, dimBlock, 0, device_stream[i]>>>(input->device_buf[i], rfloats->device_buf[i], output->device_buf[i],
                                          input->device_buf_shape[1], rfloats->device_buf_shape[1], l);

    CUDA_CALL(cudaStreamSynchronize(device_stream[i]));
  }
}

/*
 * SGEMM
 * input1: [M x K]
 * input2: [K x N]
 * output: [M x N]
 */
void matmul(Tensor *input1, Tensor *input2, Tensor *output)
{
  size_t M = input1->device_buf_shape[0];
  size_t K = input1->device_buf_shape[1];
  size_t N = input2->device_buf_shape[1];

  // For convenience, assume M is divisible by total number of processes involved
  assert(M % (mpi_size * num_devices) == 0);

  omp_set_num_threads(num_devices);

  #pragma omp parallel
  if (omp_get_num_threads() != num_devices) {
    PRINTF_WITH_RANK("Error at omp_set_num_threads\n");
    exit(1);
  }

  #pragma omp parallel for
  for (int i = 0; i < num_devices; i++) {
    int device = i;
    CUDA_CALL(cudaSetDevice(device));

    ////////////////////////////// w/o cuBLAS ////////////////////////////////////////

    dim3 dimGrid(N / TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE, M / TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE);
    dim3 dimBlock(TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE / TWO_D_BLOCKTILING_LOOP_UNROLLING_CPT, TWO_D_BLOCKTILING_LOOP_UNROLLING_TILE_SIZE / TWO_D_BLOCKTILING_LOOP_UNROLLING_RPT);

    matmul_2d_blocktiling_loop_unrolling_kernel<<<dimGrid, dimBlock, 0, device_stream[i]>>>(input1->device_buf[i], input2->device_buf[i], output->device_buf[i], 
                                        M, N, K);
    
    ////////////////////////////// w/o cuBLAS END ////////////////////////////////////////

    ////////////////////////////// w/ cuBLAS ////////////////////////////////////////

    // const float alpha = 1.0f;
    // const float beta  = 0.0f;
    // CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, input2->device_buf[i], N, input1->device_buf[i], K, &beta, output->device_buf[i], N));
    
    ////////////////////////////// w/ cuBLAS END ////////////////////////////////////////

    CUDA_CALL(cudaStreamSynchronize(device_stream[i]));
  }
}

/*
 * Optimized matrix addition kernel for CUDA
 */
__global__ void matadd_kernel(float4 *d_M, float4 *d_N, float4 *d_P, int M, int K)
{
  float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};  // temporary accumulation buffer of `WPT` items

  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;

  assert(Row < M);
  assert(Col < K / WPT);

  float4 vecM = d_M[Row * (K / WPT) + Col];
  float4 vecN = d_N[Row * (K / WPT) + Col];

  acc.x = vecM.x + vecN.x;
  acc.y = vecM.y + vecN.y;
  acc.z = vecM.z + vecN.z;
  acc.w = vecM.w + vecN.w;

  d_P[Row * (K / WPT) + Col] = acc;
}

/*
 * Optimized matrix vector addition kernel for CUDA
 */
__global__ void matvecadd_kernel(float4 *d_M, float4 *d_N, float4 *d_P, int M, int K)
{
  float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};  // temporary accumulation buffer of `WPT` items

  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;

  assert(Row < M);
  assert(Col < K / WPT);

  float4 vecM = d_M[Row * (K / WPT) + Col];
  float4 vecN = d_N[Col];

  acc.x = vecM.x + vecN.x;
  acc.y = vecM.y + vecN.y;
  acc.z = vecM.z + vecN.z;
  acc.w = vecM.w + vecN.w;

  d_P[Row * (K / WPT) + Col] = acc;
}

/*
 * Optimized matrix vector division kernel for CUDA
 */
__global__ void matvecdivide_kernel(float4 *d_M, float *d_N, float4 *d_P, int M, int K)
{
  float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};  // temporary accumulation buffer of `WPT` items

  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;

  assert(Row < M);
  assert(Col < K / WPT);

  float4 vecM = d_M[Row * (K / WPT) + Col];
  float vecN = d_N[Row];

  acc.x = vecM.x / vecN;
  acc.y = vecM.y / vecN;
  acc.z = vecM.z / vecN;
  acc.w = vecM.w / vecN;

  d_P[Row * (K / WPT) + Col] = acc;
}

/*
 * Embedding table lookup kernel for CUDA
 * 
 * in: [bsz]
 * lookup: [NUM_CHAR, K]
 * out: [bsz, K]
 */
__global__ void embedding_kernel(float* in, float4 *lookup, float4 *out, int M, int K)
{
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;

  assert(Row < M);
  assert(Col < K / WPT);

  float token = in[Row];
  out[Row * (K/WPT) + Col] = lookup[(int)token * (K / WPT) + Col];
}

/*
 * Sampling kernel for CUDA
 * 
 * IN out - [batch]
 * IN K - width of in
 * IN N - width of rfloats
 * IN l - target colum of rfloats
 */
__global__ void sampling_kernel(float *in, float *rfloats, float *out, int K, int N, int l)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  float r = rfloats[tid * N + l];
  float psum = 0.0;

  int in_offset = tid * K;

  #pragma unroll
  for (int i=0; i < K; i++) {
    psum += in[in_offset + i];
    if (psum > r) {
      out[tid] = i;
      break;
    }
  }
}

/*
 * Transpose kernel for CUDA
 * 
 * IN in - m * n matrix
 * OUT out - n * m matrix
 */
__global__ void transpose_kernel(float* in, float* out, int m, int n)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y; 

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // The shared memory `tile` is of shape [blockDim.y, blockDim.x] 
  extern __shared__ float tile[];
  int tile_width = blockDim.x;

  // Load matrix into tile
  if(col < n  && row < m){
    tile[ty*tile_width+tx] = in[row * n + col];
  }
  __syncthreads();

  unsigned int bidx, irow, icol;
  bidx = threadIdx.y * blockDim.x + threadIdx.x;
  irow = bidx / blockDim.y;
  icol = bidx % blockDim.y;

  // coordinate in transposed matrix
  row = blockIdx.x * blockDim.x + irow;
  col = blockIdx.y * blockDim.y + icol;

  // linear global memory index for transposed matrix
  unsigned int transposed_offset = row * m + col;
  __syncthreads();

  // transpose with boundary test
  if (row < n && col < m) {
      // store data to global memory from shared memory
      out[transposed_offset] = tile[icol*tile_width+irow];
  }
}

/*
 * Optimized sigmoid kernel for CUDA
 */
__global__ void sigmoid_kernel(float4 *d_M, float4 *d_P, int M, int K)
{
  float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};  // temporary accumulation buffer of `WPT` items

  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;

  assert(Row < M);
  assert(Col < K / WPT);

  float4 vecM = d_M[Row * (K / WPT) + Col];

  acc.x = 1.0 / (1.0 + expf(-vecM.x));
  acc.y = 1.0 / (1.0 + expf(-vecM.y));
  acc.z = 1.0 / (1.0 + expf(-vecM.z));
  acc.w = 1.0 / (1.0 + expf(-vecM.w));

  d_P[Row * (K / WPT) + Col] = acc;
}

/*
 * Optimized matrix hadamard product kernel for CUDA
 */
__global__ void mathadamardproduct_kernel(float4 *d_M, float4 *d_N, float4 *d_P, int M, int K)
{
  float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};  // temporary accumulation buffer of `WPT` items

  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;

  assert(Row < M);
  assert(Col < K / WPT);

  float4 vecM = d_M[Row * (K / WPT) + Col];
  float4 vecN = d_N[Row * (K / WPT) + Col];

  acc.x = vecM.x * vecN.x;
  acc.y = vecM.y * vecN.y;
  acc.z = vecM.z * vecN.z;
  acc.w = vecM.w * vecN.w;

  d_P[Row * (K / WPT) + Col] = acc;
}

/*
 * Optimized tanh kernel for CUDA
 */
__global__ void tanh_kernel(float4 *d_M, float4 *d_P, int M, int K)
{
  float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};  // temporary accumulation buffer of `WPT` items

  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;

  assert(Row < M);
  assert(Col < K / WPT);

  float4 vecM = d_M[Row * (K / WPT) + Col];

  acc.x = tanhf(vecM.x);
  acc.y = tanhf(vecM.y);
  acc.z = tanhf(vecM.z);
  acc.w = tanhf(vecM.w);

  d_P[Row * (K / WPT) + Col] = acc;
}

/*
 * Optimized (1-x) kernel for CUDA
 */
__global__ void oneminus_kernel(float4 *d_M, float4 *d_P, int M, int K)
{
  float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};  // temporary accumulation buffer of `WPT` items

  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;

  assert(Row < M);
  assert(Col < K / WPT);

  float4 vecM = d_M[Row * (K / WPT) + Col];

  acc.x = 1.0 - vecM.x;
  acc.y = 1.0 - vecM.y;
  acc.z = 1.0 - vecM.z;
  acc.w = 1.0 - vecM.w;

  d_P[Row * (K / WPT) + Col] = acc;
}


/*
 * Optimized reduction kernel for CUDA
 */
__global__ void sum_reduce_kernel(float *d_M, float *d_P, int M, int K)
{
  extern __shared__ float sdata[];

  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  sdata[ty * blockDim.x + tx] = d_M[Row * K + Col];
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s > 0; s >>= 1) {
    if (tx < s) {
      sdata[ty * blockDim.x + tx] += sdata[ty * blockDim.x + tx + s];
    }
    __syncthreads();
  }

  if (tx == 0) {
    d_P[Row] = sdata[ty * blockDim.x];
  }
}

/*
 * Optimized exp kernel for CUDA
 */
__global__ void exp_kernel(float4 *d_M, float4 *d_P, int M, int K)
{
  float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};  // temporary accumulation buffer of `WPT` items

  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;

  assert(Row < M);
  assert(Col < K / WPT);

  float4 vecM = d_M[Row * (K / WPT) + Col];

  acc.x = expf(vecM.x);
  acc.y = expf(vecM.y);
  acc.z = expf(vecM.z);
  acc.w = expf(vecM.w);

  d_P[Row * (K / WPT) + Col] = acc;
}

/*
 * Optimized memset kernel for CUDA
 */
__global__ void memset_kernel(float4 *d_M, int M, int K, float c)
{
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;

  assert(Row < M);
  assert(Col < K / WPT);

  float4 vecM = d_M[Row * (K / WPT) + Col];

  vecM.x = c;
  vecM.y = c;
  vecM.z = c;
  vecM.w = c;

  d_M[Row * (K / WPT) + Col] = vecM;
}

/*
 * Init CUDA devices and streams
 */
void init_devices()
{
  CUDA_CALL(cudaGetDeviceCount(&num_devices));

  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, i));
    if (LOG3)
      print_device_info(prop, i);

    // Assumption: Homogeneous GPUs
    CUDA_MAX_SHARED_MEM_SIZE = prop.sharedMemPerBlock;
    CUDA_MAX_THREADS_PER_BLOCK = prop.maxThreadsPerBlock;
  }

  if (num_devices <= 0) {
    PRINTF_WITH_RANK("No CUDA device found. Aborting");
    exit(1);
  }

  device_stream = (cudaStream_t*)malloc(num_devices * sizeof(cudaStream_t));

  #pragma omp parallel for num_threads(num_devices)
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaStreamCreate(&device_stream[i]));
  }
}

/*
 * Set global configs
 */
void init_configs(int names) 
{
  /* Init batch size */
  int min_bsz_per_device = TILE_SIZE;
  int max_bsz_per_device = pow(2,16); // for NVIDIA V100 32G

  int min_batch_size = min_bsz_per_device * mpi_size * num_devices;
  int max_batch_size = max_bsz_per_device * mpi_size * num_devices;

  if (names < min_batch_size) {
    BSZ = min_batch_size;
  } else if (names < max_batch_size) {
    BSZ = names;
  } else {
    BSZ = max_batch_size;
  }

  if (LOG1) {
    PRINTF_ROOT("N:%d BSZ: %d(per device:%d)\n", names, BSZ, BSZ/(mpi_size * num_devices));
  }
  assert(BSZ % (mpi_size * num_devices) == 0);
}

/* Input layer */
Tensor *batch_input;

/* Embedding layer */
Tensor *character_embedding;
Tensor *batch_emb_out;

/* GRU layers - Parameters */
Tensor *W_in0, *W_in1;
Tensor *W_hn0, *W_hn1;
Tensor *b_in0, *b_in1;
Tensor *b_hn0, *b_hn1;

/* GRU layers - Fused parameters */
Tensor *W_ir0_iz0, *W_hr0_hz0, *b_ir0_iz0, *b_hr0_hz0;
Tensor *W_ir1_iz1, *W_hr1_hz1, *b_ir1_iz1, *b_hr1_hz1;
Tensor *batch_hidden0, *batch_hidden1;

/* GRU layers - Activations */
Tensor *batch_rztmp00, *batch_rztmp01, *batch_rztmp02, *batch_rztmp03, *batch_rztmp04, *batch_rztmp05;
Tensor *batch_rztmp10, *batch_rztmp11, *batch_rztmp12, *batch_rztmp13, *batch_rztmp14, *batch_rztmp15;
Tensor *batch_ntmp00, *batch_ntmp01, *batch_ntmp02, *batch_ntmp03, *batch_ntmp04, *batch_ntmp05;
Tensor *batch_ntmp10, *batch_ntmp11, *batch_ntmp12, *batch_ntmp13, *batch_ntmp14, *batch_ntmp15;
Tensor *batch_htmp00, *batch_htmp01, *batch_htmp02;
Tensor *batch_htmp10, *batch_htmp11, *batch_htmp12;
Tensor *batch_r0, *batch_z0, *batch_n0;
Tensor *batch_r1, *batch_z1, *batch_n1;

/* Linear layer - Parameters */
Tensor *W_l, *b_l;
Tensor *batch_l;

/* Linear layer - Activations */
Tensor *batch_ltmp0;

/* Softmax layer - Activations */
Tensor *batch_reduced;

/* Sampling layer */
Tensor *batch_rfloats;
Tensor *batch_char_prob;
Tensor *batch_selected_char;
Tensor *batch_output;

/*
 * Initialize the model.
 */
void namegen_initialize(int N, int rng_seed, char *parameter_fname) {

  if (LOG1) {
    PRINTF_WITH_RANK("namegen_initialize");
  }

  init_devices();
  init_configs(N);
  omp_set_num_threads(num_devices);
  CUBLAS_CALL(cublasCreate(&handle));

  /* Only the root process reads the parameter */
  if (IS_MPI_ROOT) {
    size_t parameter_binary_size = 0;
    float *parameter =
        (float *)read_binary(parameter_fname, &parameter_binary_size);

    /* Network parameters */
    character_embedding = new Tensor({NUM_CHAR, EMBEDDING_DIM}, parameter + OFFSET0);

    W_ir0_iz0 = new Tensor({2 * HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET1);
    W_in0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET3);
    W_ir1_iz1 = new Tensor({2 * HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET4);
    W_in1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET6);

    W_hr0_hz0 = new Tensor({2 * HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET7);
    W_hn0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET9);
    W_hr1_hz1 = new Tensor({2 * HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET10);
    W_hn1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET12);

    b_ir0_iz0 = new Tensor({2 * HIDDEN_DIM}, parameter + OFFSET13);
    b_in0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET15);
    b_ir1_iz1 = new Tensor({2 * HIDDEN_DIM}, parameter + OFFSET16);
    b_in1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET18);

    b_hr0_hz0 = new Tensor({2 * HIDDEN_DIM}, parameter + OFFSET19);
    b_hn0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET21);
    b_hr1_hz1 = new Tensor({2 * HIDDEN_DIM}, parameter + OFFSET22);
    b_hn1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET24);

    W_l = new Tensor({NUM_CHAR, HIDDEN_DIM}, parameter + OFFSET25);
    b_l = new Tensor({NUM_CHAR}, parameter + OFFSET26);

    /* Transpose */
    W_ir0_iz0->transpose();
    W_hr0_hz0->transpose();
    W_in0->transpose();
    W_hn0->transpose();

    W_ir1_iz1->transpose();
    W_hr1_hz1->transpose();
    W_in1->transpose();
    W_hn1->transpose();

    W_l->transpose();

  } else {
    character_embedding = new Tensor({NUM_CHAR, EMBEDDING_DIM});

    W_ir0_iz0 = new Tensor({EMBEDDING_DIM, 2 * HIDDEN_DIM});  
    W_in0 = new Tensor({EMBEDDING_DIM, HIDDEN_DIM});
    W_ir1_iz1 = new Tensor({HIDDEN_DIM, 2 * HIDDEN_DIM});
    W_in1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM});

    W_hr0_hz0 = new Tensor({HIDDEN_DIM, 2 * HIDDEN_DIM}); 
    W_hn0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM});
    W_hr1_hz1 = new Tensor({HIDDEN_DIM, 2 * HIDDEN_DIM});
    W_hn1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM});

    b_ir0_iz0 = new Tensor({2 * HIDDEN_DIM});
    b_in0 = new Tensor({HIDDEN_DIM});
    b_ir1_iz1 = new Tensor({2 * HIDDEN_DIM});
    b_in1 = new Tensor({HIDDEN_DIM});

    b_hr0_hz0 = new Tensor({2 * HIDDEN_DIM});
    b_hn0 = new Tensor({HIDDEN_DIM});
    b_hr1_hz1 = new Tensor({2 * HIDDEN_DIM});
    b_hn1 = new Tensor({HIDDEN_DIM});

    W_l = new Tensor({HIDDEN_DIM, NUM_CHAR}); 
    b_l = new Tensor({NUM_CHAR});
  }

  /* Broadcast parameters */
  MPI_Bcast((void*)character_embedding->buf, (int)character_embedding->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Bcast((void*)W_ir0_iz0->buf, (int)W_ir0_iz0->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void*)W_in0->buf, (int)W_in0->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void*)W_ir1_iz1->buf, (int)W_ir1_iz1->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void*)W_in1->buf, (int)W_in1->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Bcast((void*)W_hr0_hz0->buf, (int)W_hr0_hz0->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void*)W_hn0->buf, (int)W_hn0->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void*)W_hr1_hz1->buf, (int)W_hr1_hz1->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void*)W_hn1->buf, (int)W_hn1->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Bcast((void*)b_ir0_iz0->buf, (int)b_ir0_iz0->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void*)b_in0->buf, (int)b_in0->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void*)b_ir1_iz1->buf, (int)b_ir1_iz1->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void*)b_in1->buf, (int)b_in1->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Bcast((void*)b_hr0_hz0->buf, (int)b_hr0_hz0->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void*)b_hn0->buf, (int)b_hn0->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void*)b_hr1_hz1->buf, (int)b_hr1_hz1->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void*)b_hn1->buf, (int)b_hn1->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Bcast((void*)W_l->buf, (int)W_l->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void*)b_l->buf, (int)b_l->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);

  /* activations */
  batch_input = new Tensor({BSZ});
  batch_emb_out = new Tensor({BSZ, EMBEDDING_DIM});

  batch_hidden0 = new Tensor({BSZ, HIDDEN_DIM});
  batch_hidden1 = new Tensor({BSZ, HIDDEN_DIM});

  batch_rztmp00 = new Tensor({BSZ, 2 * HIDDEN_DIM});
  batch_rztmp01 = new Tensor({BSZ, 2 * HIDDEN_DIM});
  batch_rztmp02 = new Tensor({BSZ, 2 * HIDDEN_DIM});
  batch_rztmp03 = new Tensor({BSZ, 2 * HIDDEN_DIM});
  batch_rztmp04 = new Tensor({BSZ, 2 * HIDDEN_DIM});
  batch_rztmp05 = new Tensor({BSZ, 2 * HIDDEN_DIM});

  batch_ntmp00 = new Tensor({BSZ, HIDDEN_DIM});
  batch_ntmp01 = new Tensor({BSZ, HIDDEN_DIM});
  batch_ntmp02 = new Tensor({BSZ, HIDDEN_DIM});
  batch_ntmp03 = new Tensor({BSZ, HIDDEN_DIM});
  batch_ntmp04 = new Tensor({BSZ, HIDDEN_DIM});
  batch_ntmp05 = new Tensor({BSZ, HIDDEN_DIM});

  batch_r0 = new Tensor({BSZ, HIDDEN_DIM});
  batch_z0 = new Tensor({BSZ, HIDDEN_DIM});
  batch_n0 = new Tensor({BSZ, HIDDEN_DIM});

  batch_htmp00 = new Tensor({BSZ, HIDDEN_DIM});
  batch_htmp01 = new Tensor({BSZ, HIDDEN_DIM});
  batch_htmp02 = new Tensor({BSZ, HIDDEN_DIM});

  batch_rztmp10 = new Tensor({BSZ, 2 * HIDDEN_DIM});
  batch_rztmp11 = new Tensor({BSZ, 2 * HIDDEN_DIM});
  batch_rztmp12 = new Tensor({BSZ, 2 * HIDDEN_DIM});
  batch_rztmp13 = new Tensor({BSZ, 2 * HIDDEN_DIM});
  batch_rztmp14 = new Tensor({BSZ, 2 * HIDDEN_DIM});
  batch_rztmp15 = new Tensor({BSZ, 2 * HIDDEN_DIM});

  batch_ntmp10 = new Tensor({BSZ, HIDDEN_DIM});
  batch_ntmp11 = new Tensor({BSZ, HIDDEN_DIM});
  batch_ntmp12 = new Tensor({BSZ, HIDDEN_DIM});
  batch_ntmp13 = new Tensor({BSZ, HIDDEN_DIM});
  batch_ntmp14 = new Tensor({BSZ, HIDDEN_DIM});
  batch_ntmp15 = new Tensor({BSZ, HIDDEN_DIM});

  batch_r1 = new Tensor({BSZ, HIDDEN_DIM});
  batch_z1 = new Tensor({BSZ, HIDDEN_DIM});
  batch_n1 = new Tensor({BSZ, HIDDEN_DIM});

  batch_htmp10 = new Tensor({BSZ, HIDDEN_DIM});
  batch_htmp11 = new Tensor({BSZ, HIDDEN_DIM});
  batch_htmp12 = new Tensor({BSZ, HIDDEN_DIM});

  batch_ltmp0 = new Tensor({BSZ, NUM_CHAR});
  batch_l = new Tensor({BSZ, NUM_CHAR});

  batch_reduced = new Tensor({BSZ});

  batch_rfloats = new Tensor({N, MAX_LEN});
  batch_char_prob = new Tensor({BSZ, NUM_CHAR});
  batch_selected_char = new Tensor({BSZ});
  batch_output = new Tensor({BSZ, MAX_LEN});

  if (LOG1) {
    PRINTF_WITH_RANK("Start device buffer allocate\n");
  }

  #pragma omp parallel for
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));

    /* Alloc device buffer */

    // Activations (w/ shape[1] == BSZ) are allocated {BSZ/(mpi_size*num_devices), ...} device buffer
    batch_input->alloc_device_buf(i, {BSZ/(mpi_size*num_devices)});
    batch_emb_out->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), EMBEDDING_DIM});
    
    batch_hidden0->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_hidden1->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});

    batch_rztmp00->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), 2 * HIDDEN_DIM});
    batch_rztmp01->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), 2 * HIDDEN_DIM});
    batch_rztmp02->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), 2 * HIDDEN_DIM});
    batch_rztmp03->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), 2 * HIDDEN_DIM});
    batch_rztmp04->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), 2 * HIDDEN_DIM});
    batch_rztmp05->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), 2 * HIDDEN_DIM});

    batch_ntmp00->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_ntmp01->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_ntmp02->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_ntmp03->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_ntmp04->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_ntmp05->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});

    batch_r0->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_z0->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_n0->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});

    batch_htmp00->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_htmp01->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_htmp02->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});

    batch_rztmp10->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), 2 * HIDDEN_DIM});
    batch_rztmp11->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), 2 * HIDDEN_DIM});
    batch_rztmp12->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), 2 * HIDDEN_DIM});
    batch_rztmp13->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), 2 * HIDDEN_DIM});
    batch_rztmp14->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), 2 * HIDDEN_DIM});
    batch_rztmp15->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), 2 * HIDDEN_DIM});

    batch_ntmp10->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_ntmp11->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_ntmp12->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_ntmp13->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_ntmp14->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_ntmp15->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});

    batch_r1->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_z1->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_n1->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});

    batch_htmp10->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_htmp11->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});
    batch_htmp12->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), HIDDEN_DIM});

    batch_ltmp0->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), NUM_CHAR});
    batch_l->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), NUM_CHAR});

    batch_reduced->alloc_device_buf(i, {BSZ/(mpi_size*num_devices)});

    batch_rfloats->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), MAX_LEN});
    batch_char_prob->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), NUM_CHAR});
    batch_selected_char->alloc_device_buf(i, {BSZ/(mpi_size*num_devices)});
    batch_output->alloc_device_buf(i, {BSZ/(mpi_size*num_devices), MAX_LEN});

    // Parameters are allocated device buffer of size == buffer
    character_embedding->alloc_device_buf(i, {NUM_CHAR, EMBEDDING_DIM});

    W_ir0_iz0->alloc_device_buf(i, {EMBEDDING_DIM, 2 * HIDDEN_DIM});
    W_in0->alloc_device_buf(i, {EMBEDDING_DIM, HIDDEN_DIM});
    W_ir1_iz1->alloc_device_buf(i, {HIDDEN_DIM, 2 * HIDDEN_DIM});
    W_in1->alloc_device_buf(i, {HIDDEN_DIM, HIDDEN_DIM});

    W_hr0_hz0->alloc_device_buf(i, {HIDDEN_DIM, 2 * HIDDEN_DIM});
    W_hn0->alloc_device_buf(i, {HIDDEN_DIM, HIDDEN_DIM});
    W_hr1_hz1->alloc_device_buf(i, {HIDDEN_DIM, 2 * HIDDEN_DIM});
    W_hn1->alloc_device_buf(i, {HIDDEN_DIM, HIDDEN_DIM});

    b_ir0_iz0->alloc_device_buf(i, {2 * HIDDEN_DIM});
    b_in0->alloc_device_buf(i, {HIDDEN_DIM});
    b_ir1_iz1->alloc_device_buf(i, {2 * HIDDEN_DIM});
    b_in1->alloc_device_buf(i, {HIDDEN_DIM});

    b_hr0_hz0->alloc_device_buf(i, {2 * HIDDEN_DIM});
    b_hn0->alloc_device_buf(i, {HIDDEN_DIM});
    b_hr1_hz1->alloc_device_buf(i, {2 * HIDDEN_DIM});
    b_hn1->alloc_device_buf(i, {HIDDEN_DIM});

    W_l->alloc_device_buf(i, {HIDDEN_DIM, NUM_CHAR});
    b_l->alloc_device_buf(i, {NUM_CHAR});

    CUDA_CALL(cudaStreamSynchronize(device_stream[i]));
  }

  if (LOG1) {
    PRINTF_WITH_RANK("End device buffer allocate\n");
  }

  #pragma omp parallel for 
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));

    /* Copy parameters to device */
    character_embedding->to(i);

    W_ir0_iz0->to(i);
    W_in0->to(i);
    W_ir1_iz1->to(i);
    W_in1->to(i);

    W_hr0_hz0->to(i);
    W_hn0->to(i);
    W_hr1_hz1->to(i);
    W_hn1->to(i);

    b_ir0_iz0->to(i);
    b_in0->to(i);
    b_ir1_iz1->to(i);
    b_in1->to(i);

    b_hr0_hz0->to(i);
    b_hn0->to(i);
    b_hr1_hz1->to(i);
    b_hn1->to(i);

    W_l->to(i);
    b_l->to(i);

    CUDA_CALL(cudaStreamSynchronize(device_stream[i]));
  }

  if (LOG1) {
    PRINTF_WITH_RANK("End device buffer set\n");
  }

  if (LOG1) {
    PRINTF_WITH_RANK("namegen_initialize exit");
  }
}

/*
 * Generate names.
 * Any input-dependent computation/communication must be done here.
 * N: # of names to generate
 * random_floats: N*MAX_LEN sequence of random floats in [0,1].
 * output: 2D-array of size N x (MAX_LEN+1), allocated at main.cpp
 */
void namegen(int N, float *random_floats, char *output) {

  if (LOG1) {
    PRINTF_WITH_RANK("namegen");
  }

  // Broadcast random_floats to batch_rfloats
  startNVTXEvent("Init-rfloats");
  if (IS_MPI_ROOT) {
    memcpy(batch_rfloats->buf, random_floats, N * MAX_LEN * sizeof(float));
  }
  MPI_Bcast((void*)batch_rfloats->buf, batch_rfloats->buf_num_elem(), MPI_FLOAT, 0, MPI_COMM_WORLD);
  stopNVTXEvent("Init-rfloats");

  startNVTXEvent("Init-output");
  if (IS_MPI_ROOT) {
    memset(output, 0, N * (MAX_LEN + 1) * sizeof(char));
  }
  stopNVTXEvent("Init-output");

  double tembedding, tgru0, tgru1, tlinear, tsoftmax, tsample; // time variables for layer-level 
  double teq; // time variables for equation-level
  double top; // time variables for operator-level

  int total_batch_steps = MAX(N/BSZ, 1);

  /* Generate N names */
  for (int b = 0; b < total_batch_steps; b++) {
    /* Initialize input and hidden vector */

    // Init first token to SOS
    startNVTXEvent("Init-1");
    device_buf_set(batch_input, SOS);
    stopNVTXEvent("Init-1");

    // Set entire batch hidden vectors to zero
    startNVTXEvent("Init-2");
    device_buf_set(batch_hidden0, 0);
    device_buf_set(batch_hidden1, 0);
    stopNVTXEvent("Init-2");

    // Copy random_floats to batch_rfloats
    startNVTXEvent("Init-3");
    copy_batch_rfloats(batch_rfloats, b);

    if (LOG2) {
      for (int i=0; i < num_devices; i++) {
        batch_rfloats->print_device_buf_info(i, "batch_rfloats");
      }
    }
    stopNVTXEvent("Init-3");

    for (int l = 0; l < MAX_LEN; l++) {

      /* Embedding layer */
      startNVTXEvent("Embedding");
      tembedding = get_time();

      embedding(batch_input, character_embedding, batch_emb_out);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_input->print_device_buf_info(i, "batch_input");
          character_embedding->print_device_buf_info(i, "character_embedding");
          batch_emb_out->print_device_buf_info(i, "batch_emb_out");
        }
      }
      print_elapsed_time(tembedding, get_time(), "embedding");
      stopNVTXEvent("Embedding");
      /* End embedding layer */

      /* GRU0 layer*/
      startNVTXEvent("GRU0");
      tgru0 = get_time();

      /* GRU0 layer r + z */
      teq = get_time();

      /*
       * batch_emb_out[BSZ, EMB_DIM] * tr(W_ir0_iz0)[EMB_DIM, 2 * HIDDEN_DIM]
       * batch_rztmp00[BSZ, 2 * HIDDEN_DIM]
       */
      startNVTXEvent("matmul & matvecadd");
      top = get_time();
      matmul(batch_emb_out, W_ir0_iz0, batch_rztmp00);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_emb_out->print_device_buf_info(i, "batch_emb_out");
          W_ir0_iz0->print_device_buf_info(i, "W_ir0_iz0");
          batch_rztmp00->print_device_buf_info(i, "batch_rztmp00");
        }
      }
      print_elapsed_time(top, get_time(), "matmul(batch_emb_out, W_ir0_iz0, batch_rztmp00)");

      /*
      * batch_rztmp00[BSZ, 2 * HIDDEN_DIM] + b_ir0_iz0[2 * HIDDEN_DIM]
      */
      top = get_time();
      matvecadd(batch_rztmp00, b_ir0_iz0, batch_rztmp02);
      stopNVTXEvent("matmul & matvecadd");

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_rztmp00->print_device_buf_info(i, "batch_rztmp00");
          b_ir0_iz0->print_device_buf_info(i, "b_ir0_iz0");
          batch_rztmp02->print_device_buf_info(i, "batch_rztmp02");
        }
      }
      print_elapsed_time(top, get_time(), "matvecadd(batch_rztmp00, b_ir0_iz0, batch_rztmp02)");

      /*
       * batch_hidden0[BSZ, HIDDEN_DIM] * tr(W_hr0_hz0)[HIDDEN_DIM, 2 * HIDDEN_DIM]
       * batch_rztmp01[BSZ, 2 * HIDDEN_DIM]
       */
      top = get_time();
      matmul(batch_hidden0, W_hr0_hz0, batch_rztmp01);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_hidden0->print_device_buf_info(i, "batch_hidden0");
          W_hr0_hz0->print_device_buf_info(i, "W_hr0_hz0");
          batch_rztmp01->print_device_buf_info(i, "batch_rztmp01");
        }
      }
      print_elapsed_time(top, get_time(), "matmul(batch_hidden0, W_hr0_hz0, batch_rztmp01)");

      /*
       * batch_rztmp01[BSZ, 2 * HIDDEN_DIM] + b_hr0_hz0[2 * HIDDEN_DIM] to all rows
       */
      top = get_time();
      matvecadd(batch_rztmp01, b_hr0_hz0, batch_rztmp03);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_rztmp01->print_device_buf_info(i, "batch_rztmp01");
          b_hr0_hz0->print_device_buf_info(i, "b_hr0_hz0");
          batch_rztmp03->print_device_buf_info(i, "batch_rztmp03");
        }
      }
      print_elapsed_time(top, get_time(), "matvecadd(batch_rztmp01, b_hr0_hz0, batch_rztmp03)");

      /*
       * batch_rztmp02[BSZ, 2 * HIDDEN_DIM] + batch_rztmp03[2 * HIDDEN_DIM]
       */
      top = get_time();
      matadd(batch_rztmp02, batch_rztmp03, batch_rztmp04);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_rztmp02->print_device_buf_info(i, "batch_rztmp02");
          batch_rztmp03->print_device_buf_info(i, "batch_rztmp03");
          batch_rztmp04->print_device_buf_info(i, "batch_rztmp04");
        }
      }
      print_elapsed_time(top, get_time(), "matadd(batch_rztmp02, batch_rztmp03, batch_rztmp04)");

      /*
       * Sigmoid batch_rztmp04
       */
      top = get_time();
      sigmoid(batch_rztmp04, batch_rztmp05);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_rztmp04->print_device_buf_info(i, "batch_rztmp04");
          batch_rztmp05->print_device_buf_info(i, "batch_rztmp05");
        }
      }
      print_elapsed_time(top, get_time(), "sigmoid(batch_rztmp04, batch_rztmp05)");

      /*
       * device_buf_copy transfer halves of batch_rztmp05 to batch_r0, batch_z0
       */
      top = get_time();
      device_buf_copy(batch_rztmp05, batch_r0, 0, 0);
      device_buf_copy(batch_rztmp05, batch_z0, 0, batch_rztmp05->device_buf_shape[1]/2);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_rztmp05->print_device_buf_info(i, "batch_rztmp05");
          batch_r0->print_device_buf_info(i, "batch_r0");
          batch_z0->print_device_buf_info(i, "batch_z0");
        }
      }
      print_elapsed_time(top, get_time(), "batch_rztmp05->device_buf_copy to batch_r0,batch_z0");

      print_elapsed_time(teq, get_time(), "GRU0 layer r + z");
      /* End GRU0 layer r + z */

      /* GRU0 layer n */
      teq = get_time();

      /*
       * batch_emb_out[BSZ, EMB_DIM] * tr(W_in0)[EMB_DIM, HIDDEN_DIM]
       * batch_ntmp00[BSZ, HIDDEN_DIM]
       */
      top = get_time();
      matmul(batch_emb_out, W_in0, batch_ntmp00);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_emb_out->print_device_buf_info(i, "batch_emb_out");
          W_in0->print_device_buf_info(i, "W_in0");
          batch_ntmp00->print_device_buf_info(i, "batch_ntmp00");
        }
      }
      print_elapsed_time(top, get_time(), "matmul(batch_emb_out, W_in0, batch_ntmp00)");

      /*
       * batch_hidden0[BSZ, HIDDEN_DIM] * tr(W_hn0)[HIDDEN_DIM, HIDDEN_DIM]
       * batch_ntmp01[BSZ, HIDDEN_DIM]
       */
      top = get_time();
      matmul(batch_hidden0, W_hn0, batch_ntmp01);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_hidden0->print_device_buf_info(i, "batch_hidden0");
          W_hn0->print_device_buf_info(i, "W_hn0");
          batch_ntmp01->print_device_buf_info(i, "batch_ntmp01");
        }
      }
      print_elapsed_time(top, get_time(), "matmul(batch_hidden0, W_hn0, batch_ntmp01)");

      /*
      * batch_ntmp00[BSZ, HIDDEN_DIM] + b_in0[HIDDEN_DIM]
      */
      top = get_time();
      matvecadd(batch_ntmp00, b_in0, batch_ntmp02);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_ntmp00->print_device_buf_info(i, "batch_ntmp00");
          b_in0->print_device_buf_info(i, "b_in0");
          batch_ntmp02->print_device_buf_info(i, "batch_ntmp02");
        }
      }
      print_elapsed_time(top, get_time(), "matvecadd(batch_ntmp00, b_in0, batch_ntmp02)");

      /*
       * batch_ntmp01[BSZ, HIDDEN_DIM] + b_hn0[HIDDEN_DIM] to all rows
       */
      top = get_time();
      matvecadd(batch_ntmp01, b_hn0, batch_ntmp03);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_ntmp01->print_device_buf_info(i, "batch_ntmp01");
          b_hn0->print_device_buf_info(i, "b_hn0");
          batch_ntmp03->print_device_buf_info(i, "batch_ntmp03");
        }
      }
      print_elapsed_time(top, get_time(), "matvecadd(batch_ntmp01, b_hn0, batch_ntmp03)");

      /*
       * batch_r0[BSZ, HIDDEN_DIM] @ batch_ntmp03[BSZ, HIDDEN_DIM] 
       */
      top = get_time();
      mathadamardproduct(batch_r0, batch_ntmp03, batch_ntmp04);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_r0->print_device_buf_info(i, "batch_r0");
          batch_ntmp03->print_device_buf_info(i, "batch_ntmp03");
          batch_ntmp04->print_device_buf_info(i, "batch_ntmp04");
        }
      }
      print_elapsed_time(top, get_time(), "mathadamardproduct(batch_r0, batch_ntmp03, batch_ntmp04)");

      /*
       * batch_ntmp02[BSZ, HIDDEN_DIM] + batch_ntmp04[BSZ, HIDDEN_DIM]
       */
      top = get_time();
      matadd(batch_ntmp02, batch_ntmp04, batch_ntmp05);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_ntmp02->print_device_buf_info(i, "batch_ntmp02");
          batch_ntmp04->print_device_buf_info(i, "batch_ntmp04");
          batch_ntmp05->print_device_buf_info(i, "batch_ntmp05");
        }
      }
      print_elapsed_time(top, get_time(), "matadd(batch_ntmp02, batch_ntmp04, batch_ntmp05)");

      /*
       * tanh batch_ntmp05
       */
      top = get_time();
      tanh(batch_ntmp05, batch_n0);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_ntmp05->print_device_buf_info(i, "batch_ntmp05");
          batch_n0->print_device_buf_info(i, "batch_n0");
        }
      }
      print_elapsed_time(top, get_time(), "tanh(batch_ntmp05, batch_n0)");

      print_elapsed_time(teq, get_time(), "GRU0 layer n");
      /* End GRU0 layer n */

      /* GRU0 layer h (hidden) */
      teq = get_time();

      /*
       * One minus batch_z0
       */
      top = get_time();
      oneminus(batch_z0, batch_htmp00);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_z0->print_device_buf_info(i, "batch_z0");
          batch_htmp00->print_device_buf_info(i, "batch_htmp00");
        }
      }
      print_elapsed_time(top, get_time(), "oneminus(batch_z0, batch_htmp00)");

      /*
       * batch_htmp00[BSZ, HIDDEN_DIM] @ batch_n0[BSZ, HIDDEN_DIM] 
       */
      top = get_time();
      mathadamardproduct(batch_htmp00, batch_n0, batch_htmp01);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_htmp00->print_device_buf_info(i, "batch_htmp00");
          batch_n0->print_device_buf_info(i, "batch_n0");
          batch_htmp01->print_device_buf_info(i, "batch_htmp01");
        }
      }
      print_elapsed_time(top, get_time(), "mathadamardproduct(batch_htmp00, batch_n0, batch_htmp01)");

      /*
       * batch_z0[BSZ, HIDDEN_DIM] @ batch_hidden0[BSZ, HIDDEN_DIM] 
       */
      top = get_time();
      mathadamardproduct(batch_z0, batch_hidden0, batch_htmp02);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_z0->print_device_buf_info(i, "batch_z0");
          batch_hidden0->print_device_buf_info(i, "batch_hidden0");
          batch_htmp02->print_device_buf_info(i, "batch_htmp02");
        }
      }
      print_elapsed_time(top, get_time(), "mathadamardproduct(batch_z0, batch_hidden0, batch_htmp02)");

      /*
       * batch_htmp01[BSZ, HIDDEN_DIM] + batch_htmp02[BSZ, HIDDEN_DIM]
       */
      top = get_time();
      matadd(batch_htmp01, batch_htmp02, batch_hidden0);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_htmp01->print_device_buf_info(i, "batch_htmp01");
          batch_htmp02->print_device_buf_info(i, "batch_htmp02");
          batch_hidden0->print_device_buf_info(i, "batch_hidden0");
        }
      }
      print_elapsed_time(top, get_time(), "matadd(batch_htmp01, batch_htmp02, batch_hidden0)");
      
      print_elapsed_time(teq, get_time(), "GRU0 layer h");
      /* End GRU0 layer h */

      print_elapsed_time(tgru0, get_time(), "GRU0");
      stopNVTXEvent("GRU0");
      /* End GRU0 layer */

      /* GRU1 layer*/
      startNVTXEvent("GRU1");
      tgru1 = get_time();

      /* GRU1 layer r + z */
      teq = get_time();

      /*
       * batch_hidden0[BSZ, HIDDEN_DIM] * tr(W_ir1_iz1)[HIDDEN_DIM, 2 * HIDDEN_DIM]
       * batch_rztmp10[BSZ, 2 * HIDDEN_DIM]
       */
      top = get_time();
      matmul(batch_hidden0, W_ir1_iz1, batch_rztmp10);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_hidden0->print_device_buf_info(i, "batch_hidden0");
          W_ir1_iz1->print_device_buf_info(i, "W_ir1_iz1");
          batch_rztmp10->print_device_buf_info(i, "batch_rztmp10");
        }
      }
      print_elapsed_time(top, get_time(), "matmul(batch_hidden0, W_ir1_iz1, batch_rztmp10)");

      /*
       * batch_hidden1[BSZ, HIDDEN_DIM] * tr(W_hr1_hz1)[HIDDEN_DIM, 2 * HIDDEN_DIM]
       * batch_rztmp11[BSZ, 2 * HIDDEN_DIM]
       */
      top = get_time();
      matmul(batch_hidden1, W_hr1_hz1, batch_rztmp11);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_hidden1->print_device_buf_info(i, "batch_hidden1");
          W_hr1_hz1->print_device_buf_info(i, "W_hr1_hz1");
          batch_rztmp11->print_device_buf_info(i, "batch_rztmp11");
        }
      }
      print_elapsed_time(top, get_time(), "matmul(batch_hidden1, W_hr1_hz1, batch_rztmp11)");

      /*
      * batch_rztmp10[BSZ, 2 * HIDDEN_DIM] + b_ir1_iz1[2 * HIDDEN_DIM]
      */
      top = get_time();
      matvecadd(batch_rztmp10, b_ir1_iz1, batch_rztmp12);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_rztmp10->print_device_buf_info(i, "batch_rztmp10");
          b_ir1_iz1->print_device_buf_info(i, "b_ir1_iz1");
          batch_rztmp12->print_device_buf_info(i, "batch_rztmp12");
        }
      }
      print_elapsed_time(top, get_time(), "matvecadd(batch_rztmp10, b_ir1_iz1, batch_rztmp12)");

      /*
       * batch_rztmp11[BSZ, 2 * HIDDEN_DIM] + b_hr1_hz1[2 * HIDDEN_DIM] to all rows
       */
      top = get_time();
      matvecadd(batch_rztmp11, b_hr1_hz1, batch_rztmp13);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_rztmp11->print_device_buf_info(i, "batch_rztmp11");
          b_hr1_hz1->print_device_buf_info(i, "b_hr1_hz1");
          batch_rztmp13->print_device_buf_info(i, "batch_rztmp13");
        }
      }
      print_elapsed_time(top, get_time(), "matvecadd(batch_rztmp11, b_hr1_hz1, batch_rztmp13)");

      /*
       * batch_rztmp12[BSZ, 2 * HIDDEN_DIM] + batch_rztmp13[2 * HIDDEN_DIM]
       */
      top = get_time();
      matadd(batch_rztmp12, batch_rztmp13, batch_rztmp14);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_rztmp12->print_device_buf_info(i, "batch_rztmp12");
          batch_rztmp13->print_device_buf_info(i, "batch_rztmp13");
          batch_rztmp14->print_device_buf_info(i, "batch_rztmp14");
        }
      }
      print_elapsed_time(top, get_time(), "matadd(batch_rztmp12, batch_rztmp13, batch_rztmp14)");

      /*
       * Sigmoid batch_rztmp14
       */
      top = get_time();
      sigmoid(batch_rztmp14, batch_rztmp15);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_rztmp14->print_device_buf_info(i, "batch_rztmp14");
          batch_rztmp15->print_device_buf_info(i, "batch_rztmp15");
        }
      }
      print_elapsed_time(top, get_time(), "sigmoid(batch_rztmp14, batch_rztmp15)");

      /*
       * device_buf_copy transfer halves of batch_rztmp15 to batch_r1, batch_z1
       */
      top = get_time();
      device_buf_copy(batch_rztmp15, batch_r1, 0, 0);
      device_buf_copy(batch_rztmp15, batch_z1, 0, batch_rztmp15->device_buf_shape[1]/2);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_rztmp15->print_device_buf_info(i, "batch_rztmp15");
          batch_r1->print_device_buf_info(i, "batch_r1");
          batch_z1->print_device_buf_info(i, "batch_z1");
        }
      }
      print_elapsed_time(top, get_time(), "batch_rztmp15->device_buf_copy to batch_r1,batch_z1");

      print_elapsed_time(teq, get_time(), "GRU1 layer r + z");
      /* End GRU1 layer r + z */

      /* GRU1 layer n */
      teq = get_time();

      /*
       * batch_hidden0[BSZ, HIDDEN_DIM] * tr(W_in1)[HIDDEN_DIM, HIDDEN_DIM]
       * batch_ntmp10[BSZ, HIDDEN_DIM]
       */
      top = get_time();
      matmul(batch_hidden0, W_in1, batch_ntmp10);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_hidden0->print_device_buf_info(i, "batch_hidden0");
          W_in1->print_device_buf_info(i, "W_in1");
          batch_ntmp10->print_device_buf_info(i, "batch_ntmp10");
        }
      }
      print_elapsed_time(top, get_time(), "matmul(batch_hidden0, W_in1, batch_ntmp10)");

      /*
       * batch_hidden1[BSZ, HIDDEN_DIM] * tr(W_hn1)[HIDDEN_DIM, HIDDEN_DIM]
       * batch_ntmp11[BSZ, HIDDEN_DIM]
       */
      top = get_time();
      matmul(batch_hidden1, W_hn1, batch_ntmp11);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_hidden1->print_device_buf_info(i, "batch_hidden1");
          W_hn1->print_device_buf_info(i, "W_hn1");
          batch_ntmp11->print_device_buf_info(i, "batch_ntmp11");
        }
      }
      print_elapsed_time(top, get_time(), "matmul(batch_hidden1, W_hn1, batch_ntmp11)");

      /*
      * batch_ntmp10[BSZ, HIDDEN_DIM] + b_in1[HIDDEN_DIM]
      */
      top = get_time();
      matvecadd(batch_ntmp10, b_in1, batch_ntmp12);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_ntmp10->print_device_buf_info(i, "batch_ntmp10");
          b_in1->print_device_buf_info(i, "b_in1");
          batch_ntmp12->print_device_buf_info(i, "batch_ntmp12");
        }
      }
      print_elapsed_time(top, get_time(), "matvecadd(batch_ntmp10, b_in1, batch_ntmp12)");

      /*
       * batch_ntmp11[BSZ, HIDDEN_DIM] + b_hn1[HIDDEN_DIM] to all rows
       */
      top = get_time();
      matvecadd(batch_ntmp11, b_hn1, batch_ntmp13);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_ntmp11->print_device_buf_info(i, "batch_ntmp11");
          b_hn1->print_device_buf_info(i, "b_hn1");
          batch_ntmp13->print_device_buf_info(i, "batch_ntmp13");
        }
      }
      print_elapsed_time(top, get_time(), "matvecadd(batch_ntmp11, b_hn1, batch_ntmp13)");

      /*
       * batch_r1[BSZ, HIDDEN_DIM] @ batch_ntmp13[BSZ, HIDDEN_DIM] 
       */
      top = get_time();
      mathadamardproduct(batch_r1, batch_ntmp13, batch_ntmp14);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_r1->print_device_buf_info(i, "batch_r1");
          batch_ntmp13->print_device_buf_info(i, "batch_ntmp13");
          batch_ntmp14->print_device_buf_info(i, "batch_ntmp14");
        }
      }
      print_elapsed_time(top, get_time(), "mathadamardproduct(batch_r1, batch_ntmp13, batch_ntmp14)");

      /*
       * batch_ntmp12[BSZ, HIDDEN_DIM] + batch_ntmp14[BSZ, HIDDEN_DIM]
       */
      top = get_time();
      matadd(batch_ntmp12, batch_ntmp14, batch_ntmp15);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_ntmp12->print_device_buf_info(i, "batch_ntmp12");
          batch_ntmp14->print_device_buf_info(i, "batch_ntmp14");
          batch_ntmp15->print_device_buf_info(i, "batch_ntmp15");
        }
      }
      print_elapsed_time(top, get_time(), "matadd(batch_ntmp12, batch_ntmp14, batch_ntmp15)");

      /*
       * tanh batch_ntmp15
       */
      top = get_time();
      tanh(batch_ntmp15, batch_n1);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_ntmp15->print_device_buf_info(i, "batch_ntmp15");
          batch_n1->print_device_buf_info(i, "batch_n1");
        }
      }
      print_elapsed_time(top, get_time(), "tanh(batch_ntmp15, batch_n1)");

      print_elapsed_time(teq, get_time(), "GRU1 layer n");
      /* End GRU1 layer n */

      /* GRU1 layer h (hidden) */
      teq = get_time();

      /*
       * One minus batch_z1
       */
      top = get_time();
      oneminus(batch_z1, batch_htmp10);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_z1->print_device_buf_info(i, "batch_z1");
          batch_htmp10->print_device_buf_info(i, "batch_htmp10");
        }
      }
      print_elapsed_time(top, get_time(), "oneminus(batch_z1, batch_htmp10)");

      /*
       * batch_htmp10[BSZ, HIDDEN_DIM] @ batch_n1[BSZ, HIDDEN_DIM] 
       */
      top = get_time();
      mathadamardproduct(batch_htmp10, batch_n1, batch_htmp11);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_htmp10->print_device_buf_info(i, "batch_htmp10");
          batch_n1->print_device_buf_info(i, "batch_n1");
          batch_htmp11->print_device_buf_info(i, "batch_htmp11");
        }
      }
      print_elapsed_time(top, get_time(), "mathadamardproduct(batch_htmp10, batch_n1, batch_htmp11)");

      /*
       * batch_z1[BSZ, HIDDEN_DIM] @ batch_hidden1[BSZ, HIDDEN_DIM] 
       */
      top = get_time();
      mathadamardproduct(batch_z1, batch_hidden1, batch_htmp12);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_z1->print_device_buf_info(i, "batch_z1");
          batch_hidden1->print_device_buf_info(i, "batch_hidden1");
          batch_htmp12->print_device_buf_info(i, "batch_htmp12");
        }
      }
      print_elapsed_time(top, get_time(), "mathadamardproduct(batch_z1, batch_hidden1, batch_htmp12)");

      /*
       * batch_htmp11[BSZ, HIDDEN_DIM] + batch_htmp12[BSZ, HIDDEN_DIM]
       */
      top = get_time();
      matadd(batch_htmp11, batch_htmp12, batch_hidden1);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_htmp11->print_device_buf_info(i, "batch_htmp11");
          batch_htmp12->print_device_buf_info(i, "batch_htmp12");
          batch_hidden1->print_device_buf_info(i, "batch_hidden1");
        }
      }
      print_elapsed_time(top, get_time(), "matadd(batch_htmp11, batch_htmp12, batch_hidden1)");
      
      print_elapsed_time(teq, get_time(), "GRU1 layer h");
      /* End GRU1 layer h */

      print_elapsed_time(tgru1, get_time(), "GRU1");
      stopNVTXEvent("GRU1");
      /* End GRU1 layer */

      /* Linear layer */
      startNVTXEvent("Linear");
      tlinear = get_time();

      /*
       * batch_hidden1[BSZ, HIDDEN_DIM] * tr(W_l)[HIDDEN_DIM, NUM_CHAR]
       * batch_ltmp0[BSZ, NUM_CHAR]
       */
      top = get_time();
      matmul(batch_hidden1, W_l, batch_ltmp0);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_hidden1->print_device_buf_info(i, "batch_hidden1");
          W_l->print_device_buf_info(i, "W_l");
          batch_ltmp0->print_device_buf_info(i, "batch_ltmp0");
        }
      }
      print_elapsed_time(top, get_time(), "matmul(batch_hidden1, W_l, batch_ltmp0)");

      /*
      * batch_ltmp0[BSZ, NUM_CHAR] + b_l[NUM_CHAR]
      */
      top = get_time();
      matvecadd(batch_ltmp0, b_l, batch_l);

      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_ltmp0->print_device_buf_info(i, "batch_ltmp0");
          b_l->print_device_buf_info(i, "b_l");
          batch_l->print_device_buf_info(i, "batch_l");
        }
      }
      print_elapsed_time(top, get_time(), "matvecadd(batch_ltmp0, b_l, batch_l)");

      print_elapsed_time(tlinear, get_time(), "linear");
      stopNVTXEvent("Linear");
      /* End linear layer */

      /* Softmax */
      startNVTXEvent("Softmax");
      tsoftmax = get_time();

      softmax(batch_l, batch_reduced, batch_char_prob);
      
      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_char_prob->print_device_buf_info(i, "batch_char_prob");
        }
      }

      print_elapsed_time(tsoftmax, get_time(), "softmax");
      stopNVTXEvent("Softmax");
      /* End softmax layer */

      /* Sampling layer */
      startNVTXEvent("Sampling");
      tsample = get_time();

      batch_sampling(batch_char_prob, batch_rfloats, batch_selected_char, l);
      if (LOG2) {
        for (int i=0; i < num_devices; i++) {
          batch_selected_char->print_device_buf_info(i, "batch_selected_char");
        }
      }

      print_elapsed_time(tsample, get_time(), "sample");
      stopNVTXEvent("Sampling");
      /* End sampling layer */

      /* Copy selected char to output & next input */
      startNVTXEvent("Post-sampling");
      #pragma omp parallel for num_threads(num_devices)
      for (int i=0; i < num_devices; i++) {
        CUDA_CALL(cudaSetDevice(i));
        
        size_t dpitch = batch_output->device_buf_shape[1] * sizeof(float);
        size_t spitch = 1 * sizeof(float);
        size_t width =  1 * sizeof(float);
        size_t height = batch_selected_char->device_buf_shape[0];

        // Copy selected char to output device buf
        CUDA_CALL(cudaMemcpy2DAsync(&batch_output->device_buf[i][l], dpitch, 
                              batch_selected_char->device_buf[i], spitch, 
                              width, height, cudaMemcpyDeviceToDevice, device_stream[i]));

        // Set next batch input
        CUDA_CALL(cudaMemcpyAsync(batch_input->device_buf[i], batch_selected_char->device_buf[i], 
                            batch_input->device_buf_num_elem() * sizeof(float), cudaMemcpyDeviceToDevice, device_stream[i]));

        CUDA_CALL(cudaStreamSynchronize(device_stream[i]));
      }
      stopNVTXEvent("Post-sampling");
      /* End copy selected char to output & next input */
    }

    if (LOG1) {
      printf("end of batch (b:%d)\n", b);
    }

    /* Gather output */

    // host copy
    startNVTXEvent("Host-copy");
    #pragma omp parallel for num_threads(num_devices)
    for (int i=0; i < num_devices; i++) {
      CUDA_CALL(cudaSetDevice(i));

      int to = batch_output->device_buf_num_elem() * (mpi_rank * num_devices + i);
      batch_output->from(i, to);

      CUDA_CALL(cudaStreamSynchronize(device_stream[i]));
    }

    if (LOG2) {
      batch_output->print_tensor_info("batch_output");
    }

    if (LOG1) {
      printf("end of host copy\n");
    }
    stopNVTXEvent("Host-copy");

    // root gather
    startNVTXEvent("Root-gather");
    int buf_send_from = batch_output->buf_num_elem() / mpi_size * mpi_rank;
    int send_cnt = batch_output->buf_num_elem() / mpi_size;
    int *recv_cnts, *displs;

    recv_cnts = (int*)malloc(sizeof(int) * mpi_size);
    displs = (int*)malloc(sizeof(int) * mpi_size);

    for (int i = 0; i < mpi_size; i++) {
      recv_cnts[i] = send_cnt;
      displs[i] = i * send_cnt;
    }

    MPI_Gatherv(&batch_output->buf[buf_send_from], send_cnt, MPI_FLOAT,
              batch_output->buf, recv_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (LOG2) {
      batch_output->print_tensor_info("batch_output(at root)");
    }
    if (LOG1) {
      printf("end of root gather\n");
    }
    stopNVTXEvent("Root-gather");

    // set `output`
    startNVTXEvent("Set-output");
    if (IS_MPI_ROOT) {
      #pragma omp parallel for num_threads(OMP_MAX_THREADS)
      for (int k = 0; k < BSZ; k++) {
        int name_uniq_id = b * BSZ + k;
        if (name_uniq_id < N) {
          for (int l = 0; l < MAX_LEN; l++) {
            output[name_uniq_id * (MAX_LEN + 1) + l] = (char)batch_output->buf[k * (MAX_LEN) + l];
          }
        }
      }
    }

    if (LOG1) {
      printf("output [%lu:%lubytes]\n", (b+1) * BSZ * (MAX_LEN+1) * sizeof(char), N*(MAX_LEN+1) * sizeof(char));
      printf("end of set output\n");
    }
    stopNVTXEvent("Set-output");
  }
}

/*
 * Finalize the model.
 * Although it is not neccessary, we recommend to deallocate and destruct
 * everything you made in namegen_initalize() and namegen().
 */
void namegen_finalize() {
  CUBLAS_CALL(cublasDestroy(handle));

  /* Input layer */
  delete batch_input;

  /* Embedding layer */
  delete character_embedding; 
  delete batch_emb_out;

  /* GRU layers - Parameters */
  delete W_in0; delete W_in1;
  delete W_hn0; delete W_hn1;
  delete b_in0; delete b_in1;
  delete b_hn0; delete b_hn1;

  /* GRU layers - Fused parameters */
  delete W_ir0_iz0; delete W_hr0_hz0; delete b_ir0_iz0; delete b_hr0_hz0;
  delete W_ir1_iz1; delete W_hr1_hz1; delete b_ir1_iz1; delete b_hr1_hz1;
  delete batch_hidden0; delete batch_hidden1;

  /* GRU layers - Activations */
  delete batch_rztmp00; delete batch_rztmp01; delete batch_rztmp02; delete batch_rztmp03; delete batch_rztmp04; delete batch_rztmp05;
  delete batch_rztmp10; delete batch_rztmp11; delete batch_rztmp12; delete batch_rztmp13; delete batch_rztmp14; delete batch_rztmp15;
  delete batch_ntmp00; delete batch_ntmp01; delete batch_ntmp02; delete batch_ntmp03; delete batch_ntmp04; delete batch_ntmp05;
  delete batch_ntmp10; delete batch_ntmp11; delete batch_ntmp12; delete batch_ntmp13; delete batch_ntmp14; delete batch_ntmp15;
  delete batch_htmp00; delete batch_htmp01; delete batch_htmp02;
  delete batch_htmp10; delete batch_htmp11; delete batch_htmp12;
  delete batch_r0; delete batch_z0; delete batch_n0;
  delete batch_r1; delete batch_z1; delete batch_n1;

  /* Linear layer - Parameters */
  delete W_l; delete b_l;
  delete batch_l;

  /* Linear layer - Activations */
  delete batch_ltmp0;

  /* Softmax layer - Activations */
  delete batch_reduced;

  /* Sampling layer */
  delete batch_rfloats;
  delete batch_char_prob;
  delete batch_selected_char;
  delete batch_output;

  #pragma omp parallel for num_threads(num_devices)
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaStreamDestroy(device_stream[i]));
  }
}