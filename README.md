# AAFFT
This is a shared memory implementation of the fast Fourier transform (FFT) on CUDA GPUs for Astro-Accelerate project.


Compile: 'make' should do that. For comparison you need fftw library and #define CHECK_USING_FFTW in debug.h.

Notes: 
1) Codes with _no_reorder do not produce correctly ordered results, thus if compared with cuFFT of fftw they would fail. This is intended as for convolution we do not need to re-order the elements. However if #define REORDER is in .cu file the FFT kernel will perform reordering operation so one can test correctness of the code.

2) This was implemented as a test if convolution code could be faster if there would be shared memory FFT code callable from kernel. thus we wanted low register count if possible in order not to burden host kernel (one which launches this FFT code) too much.


What is what (files and functions throughout implementations):
FFT*.c
Host related stuff. Allocated host memory and generate random data. Also performs checks if results are correct.


FFT-maxwell-32bit*.cu
Device related stuff + kernel. It allocates necessary memory on the device and takes care of transfers HOST <-> DEVICE. Kernel invocation is in function FFT_external_benchmark(...)
FFT_external_benchmark(
float2 d_input  - complex input (time-domain vectors)
float2 d_output - FFT result (frequency-domain vectors)
double FFT_time - for development
);

the kernel itself FFT_GPU_external(...) requires following
FFT_GPU_external(
float2 d_input  - complex input (time-domain vectors)
float2 d_output - FFT result (frequency-domain vectors)
);
this only transfers data to shared memory, call the FFT device function and then writes data back to global memory.

Function which does the FFT is 
do_FFT(
float2 s_input - shared memory with time-domain vector
);
this function perform 'in-place' FFT.

Function names might differ from implementation to implementation.

debug.h  -  controls what will code do. Not all switches work, some might not work fully. It is work in progress...
DEBUG activate/forbids printing stuff to console
CHECK activate/forbids checks for correctness of the output.
WRITE activate/forbids saving results (like execution time) to file

CUFFT activate/forbids cuFFT
EXTERNAL activate/forbids shared memory FFT
MULTIPLE activate/forbids shared memory FFT called multiple times, might not work when number of FFT to perform (first executable argument) is too low. It needs at least 100000 FFTs to be computed.

CHECK_USING_FFTW - hide/unhide parts of the code needed for comparisons, which require fftw library

params.h  	- #define FFT_LENGTH which is FFT length to be computed it must be power-of-two (256, 512, 1024, 2048)
			- #define FFT_EXP is exponent of FFT length (FFT_LENGTH=2^FFT_EXP)
timer.h - utilities
utils_cuda.h - utilities
utils_file.h - utilities

What is what (implementations):

FFT_CT_DIF is Cooley-Tukey FFT decimation in frequency code (DIF)
FFT_CT_DIT is Cooley-Tukey FFT decimation in time code (DIT)
FFT_Pease is Pease FFT DIF
FFT_Stockham is Stockham autosort FFT
FFT_Stockham_4elem is Stockham autosort FFT with 4 FFT elements calculated per thread. It is performing better but has higher register usage. Something similar (different algorithm) was used in convolution kernel.
FFT_Stockham_8elem is Stockham autosort FFT with 8 FFT elements calculated per thread. It performs even better then 4elem but register usage was too high and whole convolution kernel was slower then with 4elem.
FFT_Stockham_Radix-R is Stockham autosort FFT which can work with N=R*2^bits FFT sizes. So it is semi-non power-of-two. .cu file contain kernel for arbitrary radix, i.e. N=R^bits, but it is very slow and naive. In params.h there must be #define RADIX defined. For example 3.

Karel Adamek