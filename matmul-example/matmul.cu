#include <stdio.h> 
#include <time.h> 

#include <cublas_v2.h> 

#define EPS 1e-3 
#define CHECK_CUDA(e) \
  if ((e) != cudaSuccess) { \
    printf("[%s:%d CudaError]: %s\n", \
        __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE);                          \
  } 
#define CHECK_CUBLAS(e)                                  \
  if ((e) != CUBLAS_STATUS_SUCCESS) {                    \
    printf("[%s:%d CublasError]\n", __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                  \
  }

#define VA (4)
#define VC (4)

#if (VA == 1)
typedef float1 type_a;
#elif (VA == 2)
typedef float2 type_a;
#else
typedef float4 type_a;
#endif

#if (VC == 1)
typedef float1 type_c;
#elif (VC == 2)
typedef float2 type_c;
#else
typedef float4 type_c;
#endif

#define MAX(a, b) (((a) < (b)) ? (b) : (a))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 32;
constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 16;
constexpr int REG_M = ((TILE_M + BLOCK_M - 1) / BLOCK_M); // 한 스레드가 output tile에서 세로 방향에서 처리하는 (vector) 개수
constexpr int REG_N = (((TILE_N / VC) + BLOCK_N - 1) / BLOCK_N);  // 한 스레드가 output tile에서 가로 방향에서 처리하는 vector 개수
constexpr int A_N = MAX(MIN(TILE_K / VA, BLOCK_N), 1);  // A tile 한 row load 당 thread 개수 (row 사이즈보다 작은 경우, 여러 차례에 나눠 진행)
constexpr int A_M = (BLOCK_M * BLOCK_N) / A_N;  // 한 threadblock이 한번에 load하는 A tile의 row 개수
constexpr int B_N = MAX(MIN(TILE_N / VC, BLOCK_N), 1);  // B tile 한 row load 당 thread의 개수
constexpr int B_M = (BLOCK_M * BLOCK_N) / B_N;  // 한 threadblock이 한번에 load하는 B tile의 row 개수
constexpr int TILE_K_VA = TILE_K / VA;
constexpr int TILE_N_VC = TILE_N / VC;

#define WARMUP 

__global__ void kernel_matmul_t_opt(type_a* A, type_c* B, type_c* C, int M, int N, int K) {
  int _K = K / VA;
  int _N = N / VC;
  int C_N = N / VC;

  if (blockIdx.x * TILE_N >= N || blockIdx.y * TILE_M >= M) return;

  __shared__ type_a Ashared[TILE_M][TILE_K / VA];
  __shared__ type_c Bshared[TILE_K][TILE_N / VC];

  type_a ZEROA = { 0.f };
  type_c ZEROB = { 0.f };
  type_c ZEROC = { 0.f };

  type_c creg[REG_M][REG_N];
  for (int y = 0; y < REG_M; ++y) {
    for (int x = 0; x < REG_N; ++x) {
      creg[y][x] = ZEROC;
    }
  }

  int ax = (threadIdx.y * blockDim.x + threadIdx.x) % A_N;  // A tile load 시 (여러번 걸릴 수 있음) thread 당 x좌표 (여러번하면 움직임)
  int ay = (threadIdx.y * blockDim.x + threadIdx.x) / A_N;  // A tile load 시 (여러번 걸릴 수 있음) thread 당 y좌표

  int bx = (threadIdx.y * blockDim.x + threadIdx.x) % B_N; // B tile load 시 (여러번 걸릴 수 있음) thread 당 x좌표
  int by = (threadIdx.y * blockDim.x + threadIdx.x) / B_N; // B tile load 시 (여러번 걸릴 수 있음) thread 당 y좌표


  for (int tk = 0; tk < K; tk += TILE_K) {

#pragma unroll
    for (int ii = 0; ii < TILE_M / A_M; ++ii) { // A tile의 가로-wise 여러번 걸림
      int li = A_M * ii + ay;
      int Ai = blockIdx.y * TILE_M + li;
#pragma unroll
      for (int kk = 0; kk < TILE_K_VA / A_N; ++kk) {  // A tile의 세로-wise (한 row) 여러번 걸림 
        int lj = A_N * kk + ax;
        int Aj = (tk / VA) + lj;
        type_a val = (Ai < M && Aj < _K) ? A[Ai * _K + Aj] : ZEROA;
        Ashared[li][lj] = val;
      }
    }

#pragma unroll
    for (int kk = 0; kk < TILE_K / B_M; ++kk) {
      int li = B_M * kk + by;
      int Bi = tk + li;
#pragma unroll
      for (int jj = 0; jj < TILE_N_VC / B_N; ++jj) {
        int lj = B_N * jj + bx;
        int Bj = blockIdx.x * TILE_N_VC + lj;
        type_c val = (Bi < K && Bj < _N) ? B[Bi * _N + Bj] : ZEROB;
        Bshared[li][lj] = val;
      }
    }

    __syncthreads();

#pragma unroll
    for (int y = 0; y < REG_M; ++y) {
      int si = threadIdx.y + y * BLOCK_M; // 내가 계산할 A tile의 y좌표
#pragma unroll
      for (int x = 0; x < REG_N; ++x) {
        int sj = threadIdx.x + x * BLOCK_N; // 내가 계산할 B tile의 x좌표
#pragma unroll
        for (int lk = 0; lk < TILE_K / VA; ++lk) {
          // A tile의 한 원소는 벡터 (4개) -> 한 원소는 B 타일의 x좌표에 해당하는 곳에서 (가로) 위부터 가로로 4개씩 곱해서 creg로 더함 

          creg[y][x].x += Ashared[si][lk].x * Bshared[VC * lk + 0][sj].x;
          creg[y][x].y += Ashared[si][lk].x * Bshared[VC * lk + 0][sj].y;
          creg[y][x].z += Ashared[si][lk].x * Bshared[VC * lk + 0][sj].z;
          creg[y][x].w += Ashared[si][lk].x * Bshared[VC * lk + 0][sj].w;

          creg[y][x].x += Ashared[si][lk].y * Bshared[VC * lk + 1][sj].x;
          creg[y][x].y += Ashared[si][lk].y * Bshared[VC * lk + 1][sj].y;
          creg[y][x].z += Ashared[si][lk].y * Bshared[VC * lk + 1][sj].z;
          creg[y][x].w += Ashared[si][lk].y * Bshared[VC * lk + 1][sj].w;

          creg[y][x].x += Ashared[si][lk].z * Bshared[VC * lk + 2][sj].x;
          creg[y][x].y += Ashared[si][lk].z * Bshared[VC * lk + 2][sj].y;
          creg[y][x].z += Ashared[si][lk].z * Bshared[VC * lk + 2][sj].z;
          creg[y][x].w += Ashared[si][lk].z * Bshared[VC * lk + 2][sj].w;

          creg[y][x].x += Ashared[si][lk].w * Bshared[VC * lk + 3][sj].x;
          creg[y][x].y += Ashared[si][lk].w * Bshared[VC * lk + 3][sj].y;
          creg[y][x].z += Ashared[si][lk].w * Bshared[VC * lk + 3][sj].z;
          creg[y][x].w += Ashared[si][lk].w * Bshared[VC * lk + 3][sj].w;

        }
      }
    }
    __syncthreads();
  }

// 내가 처리해야 할 creg 값 16개 처리
#pragma unroll
  for (int y = 0; y < REG_M; ++y) {
#pragma unroll
    for (int x = 0; x < REG_N; ++x) {
      int i = blockIdx.y * TILE_M + threadIdx.y + y * BLOCK_M;
      int j = blockIdx.x * (TILE_N / VC) + threadIdx.x + x * BLOCK_N;
      C[i * C_N + j] = creg[y][x];
    }
  }
}

int main(int argc, char* argv[])
{
  int M_, N_, K_;


  M_ = 4096;
  N_ = 4096;
  K_ = 4096;

  float* a = (float*)malloc(sizeof(float) * M_ * K_);
  float* b = (float*)malloc(sizeof(float) * N_ * K_);
  float* c = (float*)malloc(sizeof(float) * M_ * N_);
  float* c_ans = (float*)malloc(sizeof(float) * M_ * N_);

  float* d_a;
  float* d_b;
  float* d_c;
  float* d_c_ans;
  CHECK_CUDA(cudaMalloc(&d_a, sizeof(float) * M_ * K_));
  CHECK_CUDA(cudaMalloc(&d_b, sizeof(float) * N_ * K_));
  CHECK_CUDA(cudaMalloc(&d_c, sizeof(float) * M_ * N_));
  CHECK_CUDA(cudaMalloc(&d_c_ans, sizeof(float) * M_ * N_));


  for (int i = 0; i < M_ * K_; ++i) {
    a[i] = 2 * (rand() / (double)RAND_MAX);
  }
  for (int i = 0; i < N_ * K_; ++i) {
    b[i] = 2 * (rand() / (double)RAND_MAX);
  }

  CHECK_CUDA(cudaMemcpy(d_a, a, sizeof(float) * M_ * K_, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b, sizeof(float) * N_ * K_, cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  printf("Running kernel\n");
#ifdef WARMUP
  {
    for (int ii = 0; ii < 10; ++ii) {
      dim3 threadDims = { BLOCK_N, BLOCK_M, 1 };
      dim3 blockDims = {
        (unsigned int)((N_ + TILE_N - 1) / (TILE_N)),
        (unsigned int)((M_ + TILE_M - 1) / (TILE_M)),
        1,
      };
      kernel_matmul_t_opt << <blockDims, threadDims >> > (
        (type_a*)d_a,
        (type_c*)d_b,
        (type_c*)d_c,
        M_,
        N_,
        K_
        );
      CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
  }
#endif

  struct timespec s, e;
  clock_gettime(CLOCK_MONOTONIC, &s);
  {
    dim3 threadDims = { BLOCK_N, BLOCK_M, 1 };
    dim3 blockDims = {
      (unsigned int)((N_ + TILE_N - 1) / (TILE_N)),
      (unsigned int)((M_ + TILE_M - 1) / (TILE_M)),
      1,
    };

    kernel_matmul_t_opt << <blockDims, threadDims >> > (
      (type_a*)d_a,
      (type_c*)d_b,
      (type_c*)d_c,
      M_,
      N_,
      K_
      );
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());
  }

  clock_gettime(CLOCK_MONOTONIC, &e);
  double elapsed = (e.tv_sec - s.tv_sec) + ((double)e.tv_nsec - s.tv_nsec) / 1000000000.;
  double bw = 2.0 * M_ * K_ * N_ / 1000000000. / elapsed;
  printf("elapsed time: %lfs, bandwidth: %lf GB/s\n", elapsed, bw);
  CHECK_CUDA(cudaMemcpy(c, d_c, sizeof(float) * M_ * N_, cudaMemcpyDeviceToHost));

  if (argc == 2) {
    struct timespec s, e;
    clock_gettime(CLOCK_MONOTONIC, &s);
    {
      printf("Running cublas\n");
      float alpha = 1.f;
      float beta = 0.f;
      CHECK_CUBLAS(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N_, M_, K_,
        &alpha,
        d_b, K_,
        d_a, K_,
        &beta,
        d_c_ans, N_)
      );
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    clock_gettime(CLOCK_MONOTONIC, &e);
    double elapsed = (e.tv_sec - s.tv_sec) + ((double)e.tv_nsec - s.tv_nsec) / 1000000000.;
    double cublas_bw = 2.0 * M_ * K_ * N_ / 1000000000. / elapsed;
    printf("elapsed time: %lfs, bandwidth: %lf GB/s\n", elapsed, cublas_bw);

    printf("Kernel / cuBlas = %lf / %lf = %lf %%\n",
      bw, cublas_bw, bw / cublas_bw * 100);
    CHECK_CUDA(cudaMemcpy(c_ans, d_c_ans, sizeof(float) * M_ * N_, cudaMemcpyDeviceToHost));

    for (int i = 0; i < M_; ++i) {
      for (int j = 0; j < N_; ++j) {
        if (fabs((c[i * N_ + j] - c_ans[i * N_ + j]) / c[i * N_ + j]) >= EPS) {
          printf("Validation Failed! C[%d, %d]: %f %f\n", i, j, c[i * N_ + j], c_ans[i * N_ + j]);
          exit(1);
        }
      }
    }
    printf("Verification Success!\n");
  }
  return 0;
}
