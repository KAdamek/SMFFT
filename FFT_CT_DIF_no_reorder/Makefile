###############################################################
# CUDA_HOME are supposed to be on default position
# and set it in your PATH .bashrc
###############################################################
INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcufft -lcuda

GCC = g++
NVCC = ${CUDA_HOME}/bin/nvcc

NVCCFLAGS = -O3 -arch=sm_61 --ptxas-options=-v --use_fast_math -Xcompiler -Wextra -lineinfo

GCC_OPTS =-O3 -Wall -Wextra $(INC)

ANALYZE = FFT_DIF.exe

ifdef reglim
NVCCFLAGS += --maxrregcount=$(reglim)
endif

ifdef fftw
LIB += -lfftw3f
endif

all: clean analyze

analyze: FFT_DIF.o FFT_DIF-maxwell-32bit.o Makefile
	$(NVCC) -o $(ANALYZE) FFT_DIF-maxwell-32bit.o FFT_DIF.o $(LIB) $(NVCCFLAGS) 

FFT_DIF-maxwell-32bit.o: timer.h utils_cuda.h utils_file.h
	$(NVCC) -c FFT_DIF-maxwell-32bit.cu $(NVCCFLAGS)

FFT_DIF.o: FFT_DIF.c
	$(GCC) -c FFT_DIF.c $(GCC_OPTS)

reference.o: reference.c 
	$(GCC) -c reference.c $(GCC_OPTS) $(INC) -lfftw3

clean:	
	rm -f *.o *.~ $(ANALYZE)


