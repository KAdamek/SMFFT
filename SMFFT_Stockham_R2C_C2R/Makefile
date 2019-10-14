INC := -I/usr/local/cuda/include
LIB := -L/usr/local/cuda/lib64 -lcudart -lcufft -lfftw3f -lcuda

# use this compilers
# g++ just because the file write
GCC = g++
NVCC = nvcc


###############################################################
# Basic flags for compilers, one for debug options
# fmad flags used for reason of floating point operation
###############################################################
NVCCFLAGS = -O2 -arch=sm_70 --ptxas-options=-v --use_fast_math -Xcompiler -Wextra -lineinfo

GCC_OPTS =-O3 -Wall -Wextra $(INC)

ANALYZE = FFT.exe

all: clean analyze

analyze: FFT.o FFT-maxwell-32bit-Stockham.o Makefile
	$(NVCC) -o $(ANALYZE) FFT-maxwell-32bit-Stockham.o FFT.o $(LIB) $(NVCCFLAGS) 

FFT-maxwell-32bit-Stockham.o: timer.h utils_cuda.h
	$(NVCC) -c FFT-maxwell-32bit-Stockham.cu $(NVCCFLAGS)

FFT.o: FFT.c
	$(GCC) -c FFT.c $(GCC_OPTS)

clean:	
	rm -f *.o *.~ $(ANALYZE)


