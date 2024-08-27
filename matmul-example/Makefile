matmul: matmul.cu
	nvcc -O3 $< -o $@ -lcublas

clean:
	rm -rf matmul
