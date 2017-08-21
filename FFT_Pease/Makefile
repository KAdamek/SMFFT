<<<<<<< HEAD
INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcufft -lcuda

GCC = g++
NVCC = ${CUDA_HOME}/bin/nvcc

NVCCFLAGS = -O3 -arch=sm_61 --ptxas-options=-v --use_fast_math -Xcompiler -Wextra -lineinfo
=======
INC := -I/usr/local/cuda/include
LIB := -L/usr/local/cuda/lib64 -lcudart -lcufft -lcuda -lgsl -lgslcblas -lfftw3f

GCC = g++
NVCC = nvcc

NVCCFLAGS = -O3 -arch=sm_52 --ptxas-options=-v --use_fast_math -Xcompiler -Wextra -lineinfo
>>>>>>> origin/master

GCC_OPTS =-O3 -Wall -Wextra $(INC)

EXECUTABLE = FFT.exe

<<<<<<< HEAD
ifdef reglim
NVCCFLAGS += --maxrregcount=$(reglim)
endif

ifdef fftw
LIB += -lfftw3f
endif
=======
all: clean executable
>>>>>>> origin/master

executable: FFT.o FFT-maxwell-32bit-Pease.o Makefile
	$(NVCC) -o $(EXECUTABLE) FFT-maxwell-32bit-Pease.o FFT.o $(LIB) $(NVCCFLAGS) 

FFT-maxwell-32bit-Pease.o: timer.h utils_cuda.h utils_file.h
	$(NVCC) -c FFT-maxwell-32bit-Pease.cu $(NVCCFLAGS)

FFT.o: FFT.c
	$(GCC) -c FFT.c $(GCC_OPTS)

reference.o: reference.c 
	$(GCC) -c reference.c $(GCC_OPTS) $(INC) -lfftw3

clean:	
	rm -f *.o *.~ $(EXECUTABLE)


