INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcufft -lcuda

GCC = g++
NVCC = ${CUDA_HOME}/bin/nvcc

NVCCFLAGS = -O3 -arch=sm_70 --ptxas-options=-v --use_fast_math -Xcompiler -Wextra -lineinfo

GCC_OPTS =-O3 -Wall -Wextra $(INC)

ANALYZE = FFT.exe

all: clean analyze

analyze: FFT.o FFT-GPU-32bit.o Makefile
	$(NVCC) -o $(ANALYZE) FFT-GPU-32bit.o FFT.o $(LIB) $(NVCCFLAGS) 

FFT-GPU-32bit.o: timer.h utils_cuda.h
	$(NVCC) -c FFT-GPU-32bit.cu $(NVCCFLAGS)

FFT.o: FFT.c
	$(GCC) -c FFT.c $(GCC_OPTS)

clean:	
	rm -f *.o *.~ $(ANALYZE)


