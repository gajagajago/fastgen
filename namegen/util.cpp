#include "util.h"

#include <cstdio>
#include <cstdlib>
#include <mpi.h>

// Defined in main.cpp
extern int mpi_rank, mpi_size;

void *read_binary(const char *filename, size_t *size) {
  size_t size_;
  FILE *f = fopen(filename, "rb");
  CHECK_ERROR(f != NULL, "Failed to read %s", filename);
  fseek(f, 0, SEEK_END);
  size_ = ftell(f);
  rewind(f);
  void *buf = malloc(size_);
  size_t ret = fread(buf, 1, size_, f);
  fclose(f);
  CHECK_ERROR(size_ == ret, "Failed to read %ld bytes from %s", size_,
              filename);
  if (size != NULL)
    *size = size_;
  return buf;
}

void WriteFile(const char *filename, size_t size, void *buf) {
  FILE *f = fopen(filename, "wb");
  CHECK_ERROR(f != NULL, "Failed to write %s", filename);
  size_t ret = fwrite(buf, 1, size, f);
  fclose(f);
  CHECK_ERROR(size == ret, "Failed to write %ld bytes to %s", size, filename);
}

double get_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

void print_mat(float *m, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%+.3f ", m[i * N + j]);
    }
    printf("\n");
  }
}

void print_vec_int(int *m, int M) {
  for (int i = 0; i < M; ++i) {
    printf("%d ", m[i]);
  }
  printf("\n");
}

void print_device_array(int device, const char* name, float* d_array, int cnt)
{
  int i;
  printf("\n\n");
  printf("%s[%d]\n", name, device);

  float* h_array = (float*)malloc(sizeof(float) * cnt);
  cudaMemcpy(h_array, d_array, cnt * sizeof(float), cudaMemcpyDeviceToHost);

  printf("\t");
  for (i = 0; i < 3; i++) {
    printf("%.3f ", h_array[i]);
  }
  printf("\n");

  free(h_array);
}

/*
 * Print GPU information 
 * 
 * IN prop - Device property
 * IN rank - Device rank
 */
void print_device_info(cudaDeviceProp prop, int rank)
{
  printf("(%d)"
  "\tName: %s"
  "\n"
  "\tMax registers per block: %d"
  "\tMax registers per SM: %d"
  "\n" 
  "\tMax threads per block: %d"
  "\tMax threads per SM: %d"
  "\n"
  "\tShared mem per block: %lu"
  "\tShared mem per SM: %lu"
  "\n"
  "\tThreads per warp: %d"
  "\tMax warps per SM: %d"
  "\n", 
  rank,
  prop.name, 
  prop.regsPerBlock,
  prop.regsPerMultiprocessor,
  prop.maxThreadsPerBlock,
  prop.maxThreadsPerMultiProcessor,
  prop.sharedMemPerBlock,
  prop.sharedMemPerMultiprocessor,
  prop.warpSize,
  prop.maxThreadsPerMultiProcessor/prop.warpSize);
}