INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcufft -lcuda

GCC = g++
NVCC = ${CUDA_HOME}/bin/nvcc

NVCCFLAGS = -O3 -arch=sm_61 --ptxas-options=-v --use_fast_math -Xcompiler -Wextra -lineinfo

GCC_OPTS =-O3 -Wall -Wextra $(INC)

ANALYZE = FFT_DIT.exe

ifdef reglim
NVCCFLAGS += --maxrregcount=$(reglim)
endif

ifdef fftw
LIB += -lfftw3f
endif

all: clean analyze

analyze: FFT_DIT.o FFT_DIT-maxwell-32bit.o Makefile
	$(NVCC) -o $(ANALYZE) FFT_DIT-maxwell-32bit.o FFT_DIT.o $(LIB) $(NVCCFLAGS) 

FFT_DIT-maxwell-32bit.o: timer.h utils_cuda.h utils_file.h
	$(NVCC) -c FFT_DIT-maxwell-32bit.cu $(NVCCFLAGS)

FFT_DIT.o: FFT_DIT.c
	$(GCC) -c FFT_DIT.c $(GCC_OPTS)

reference.o: reference.c 
	$(GCC) -c reference.c $(GCC_OPTS) $(INC) -lfftw3

clean:	
	rm -f *.o *.~ $(ANALYZE)


