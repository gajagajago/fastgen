TARGET=main
OBJECTS=main.o util.o namegen.o matmul-kernel.o

CFLAGS=-std=c++14 -O3 -Wall -march=native -mavx2 -mfma -mno-avx512f -fopenmp -I/usr/local/cuda/include
CUDA_CFLAGS:=$(foreach option, $(CFLAGS),-Xcompiler=$(option)) -arch=sm_70 -gencode=arch=compute_70,code=sm_70

LDFLAGS=-pthread -L/usr/local/cuda/lib64
LDLIBS=-lmpi_cxx -lmpi -lstdc++ -lcudart -lcublas -lm -lnvToolsExt

CXX=g++
CUX=/usr/local/cuda/bin/nvcc

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CFLAGS) -c -o $@ $^

%.o: %.cu
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)

