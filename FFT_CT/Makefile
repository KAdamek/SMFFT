INC := -I/usr/local/cuda/include
LIB := -L/usr/local/cuda/lib64 -lcudart -lcufft -lfftw3f -lcuda

GCC = g++
NVCC = nvcc


NVCCFLAGS = -O2 -arch=sm_52 --ptxas-options=-v --use_fast_math -Xcompiler -Wextra -lineinfo

GCC_OPTS =-O3 -Wall -Wextra $(INC)

ANALYZE = FFT.exe

all: clean analyze

analyze: FFT.o FFT-maxwell-32bit.o Makefile
	$(NVCC) -o $(ANALYZE) FFT-maxwell-32bit.o FFT.o $(LIB) $(NVCCFLAGS) 

FFT-maxwell-32bit.o: timer.h utils_cuda.h utils_file.h
	$(NVCC) -c FFT-maxwell-32bit.cu $(NVCCFLAGS)

FFT.o: FFT.c
	$(GCC) -c FFT.c $(GCC_OPTS)

reference.o: reference.c 
	$(GCC) -c reference.c $(GCC_OPTS) $(INC) -lfftw3

clean:	
	rm -f *.o *.~ $(ANALYZE)


