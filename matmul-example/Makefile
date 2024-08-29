matmul: matmul.cu
	nvcc -O3 $< -o $@ -lcublas

matmul_column: matmul_column.cu
	nvcc -O3 $< -o $@ -lcublas

clean:
	rm -rf matmul matmul_half matmul_column
