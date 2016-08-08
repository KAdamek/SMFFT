INC := -I/usr/local/cuda/include
LIB := -L/usr/local/cuda/lib64 -lcudart -lcufft -lcuda -lgsl -lgslcblas -lfftw3f

GCC = g++
NVCC = nvcc

NVCCFLAGS = -O3 -arch=sm_52 --ptxas-options=-v --use_fast_math -Xcompiler -Wextra -lineinfo

GCC_OPTS =-O3 -Wall -Wextra $(INC)

EXECUTABLE = FFT.exe

all: clean executable

executable: FFT.o FFT-maxwell-32bit-Stockham.o Makefile
	$(NVCC) -o $(EXECUTABLE) FFT-maxwell-32bit-Stockham.o FFT.o $(LIB) $(NVCCFLAGS) 

FFT-maxwell-32bit-Stockham.o: timer.h utils_cuda.h utils_file.h
	$(NVCC) -c FFT-maxwell-32bit-Stockham.cu $(NVCCFLAGS)

FFT.o: FFT.c
	$(GCC) -c FFT.c $(GCC_OPTS)

reference.o: reference.c 
	$(GCC) -c reference.c $(GCC_OPTS) $(INC) -lfftw3

clean:	
	rm -f *.o *.~ $(EXECUTABLE)


