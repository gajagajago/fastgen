// Copyright (c) 2023-present, Junyeol Ryu

#include "matmul.h"
#include "util.h"
#include "matmul-kernel.h"

#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*f)();

static bool print_matrix = false;
static bool validation = false;
static bool print_device = false;
int M = 8;
int N = 8;
int K = 8;
static int num_iterations = 1;

static void print_help(const char *prog_name) {
  printf("Usage: %s [-pvd] [-n num_iterations] M N K\n", prog_name);
  printf("Options:\n");
  printf("     -p : print matrix. (default: off)\n");
  printf("     -v : validate matmul. (default: off)\n");
  printf("     -d : print device info. (default: off)\n");
  printf("     -n : number of iterations (default: 1)\n");
  printf("      M : number of rows of matrix A and C. (default: 8)\n");
  printf("      N : number of columns of matrix B and C. (default: 8)\n");
  printf(
      "      K : number of columns of matrix A and rows of B. (default: 8)\n");
}

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "pvdt:n:m:")) != -1) {
    switch (c) {
    case 'p':
      print_matrix = true;
      break;
    case 'v':
      validation = true;
      break;
      break;
    case 'n':
      num_iterations = atoi(optarg);
      break;
    case 'd':
      print_device = true;
      break;
    default:
      print_help(argv[0]);
      exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
    case 0:
      M = atoi(argv[i]);
      break;
    case 1:
      N = atoi(argv[i]);
      break;
    case 2:
      K = atoi(argv[i]);
      break;
    default:
      break;
    }
  }

  printf("Options:\n");
  printf("  Problem size: M = %d, N = %d, K = %d\n", M, N, K);
  printf("  Number of iterations: %d\n", num_iterations);
  printf("  Print matrix: %s\n", print_matrix ? "on" : "off");
  printf("  Print device: %s\n", print_device ? "on" : "off");
  printf("  Validation: %s\n", validation ? "on" : "off");
  printf("\n");
}

int main(int argc, char **argv) {

  parse_opt(argc, argv);
  fflush(stdout);

  if (print_device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    print_device_info(prop);
  }

  int num_kernels = 1;
  f kernel_f[num_kernels] = {
    // &matmul_cublas, 
    // &matmul_naive,
    // &matmul_global_mem_coalescing,
    // &matmul_shared_mem,
    // &matmul_1d_blocktiling_naive,
    // &matmul_1d_blocktiling_row,
    // &matmul_1d_blocktiling_row_vector_type,
    // &matmul_1d_blocktiling_column,
    // &matmul_2d_blocktiling,
    &matmul_2d_blocktiling_loop_unrolling,
  };

  for (int k=0; k < num_kernels; k++) {

    printf("\n"
    "==============Kernel %d=============="
    "\n", k);
    
    timer_init();

    float *A, *B, *C;
    alloc_mat(&A, M, K);
    alloc_mat(&B, K, N);
    alloc_mat(&C, M, N);
    printf("Initializing matrices...");
    rand_mat(A, M, K);
    rand_mat(B, K, N);
    printf("Done!\n");

    matmul_initialize(A, B);

    // Few warmup iterations
    zero_mat(C, M, N);
    for (int i = 0; i < 3; i++) {
      kernel_f[k]();
    }

    double elapsed_time_sum = 0;
    for (int i = 0; i < num_iterations; ++i) {
      printf("Calculating...(iter=%d) ", i);
      fflush(stdout);
      zero_mat(C, M, N);

      timer_start(0);
      kernel_f[k]();
      double elapsed_time = timer_stop(0);

      printf("%f sec\n", elapsed_time);
      elapsed_time_sum += elapsed_time;
    }

    matmul_finalize(C);

    if (print_matrix) {
      printf("MATRIX A:\n");
      print_mat(A, M, K);
      printf("MATRIX B:\n");
      print_mat(B, K, N);
      printf("MATRIX C:\n");
      print_mat(C, M, N);
    }

    if (validation) {
      check_mat_mul(A, B, C, M, N, K);
    }

    double elapsed_time_avg = elapsed_time_sum / num_iterations;
    printf("Avg. time: %f sec\n", elapsed_time_avg);
    printf("Avg. throughput: %f GFLOPS\n",
            2.0 * M * N * K / elapsed_time_avg / 1e9);
  }

  return 0;
}
