# AAFFT
This is a shared memory implementation of the fast Fourier transform (FFT) on CUDA GPUs for Astro-Accelerate project.


Compile: 'make' should do that. It requires fftw library for comparisons. This dependency could be removed from FFT.c if desired.

Notes: 
1) Codes with _no_reorder do not produce correctly ordered results, thus if compared with cuFFT of fftw they would fail. This is intended as for convolution we do not need to re-order the elements.

2) This was implemented as a test if convolution code could be faster if there would be FFT code callable from kernel. Thus this FFT implementation was aimed to be more flexible in sense that I avoided any FFT size dependent code, i.e. if N=1024 run this or that. I intended to try to optimize for final FFT size but it never came to that.

3) Also we wanted low register count if possible in order not to burden host kernel (one which launches this FFT code) too much.

4) There is, I think, a room for optimizations. For example more reuse of computed twiddle factors, try to use Radix-4, 8, ... for calculation of N=power-of-two. Higher radices would allow use of #defined values of twiddle factors more easily than current solution. Pre-compute twiddle factors for given FFT size N. 


What is what (files and functions throughout implementations):
FFT.c
Host related stuff. Allocated host memory and generate random data. Also performs checks if results are correct.


FFT-maxwell-32bit*.cu
Device related stuff + kernel. It allocates necessary memory on the device and takes care of transfers HOST <-> DEVICE. Kernel invocation is in function FFT_external_benchmark(...)
FFT_external_benchmark(
float2 d_input  - complex input (time-domain vectors)
float2 d_output - FFT result (frequency-domain vectors)
int nSamples    - FFT size (N)
int nSpectra    - number of time-domain vectors we want to be FFT-ied
double FFT_time - for development
);

the kernel itself FFT_GPU_external(...) requires following
FFT_GPU_external(
float2 d_input  - complex input (time-domain vectors)
float2 d_output - FFT result (frequency-domain vectors)
int N           - FFT size
int bits        - it is an exponent of radix base, i.e. 2^bits=N
);
this only transfers data to shared memory, call the FFT device function and then writes data back to global memory.

Function which does the FFT is 
do_FFT(
float2 s_input - shared memory with time-domain vector
int N          - FFT size
int bits       - it is an exponent of radix base, i.e. 2^bits=N
);
this function perform 'in-place' FFT.

Function names might differ from implementation to implementation.

debug.h  -  controls what will code do. Not all switches work, some might not work fully. It is work in progress...
DEBUG activate/forbids printing stuff to console
CHECK activate/forbids checks for correctness of the output.
WRITE activate/forbids saving results (like execution time) to file
CUFFT activate/forbids cuFFT
INTERNAL does not work
EXTERNAL activate/forbids shared memory FFT
MULTIPLE activate/forbids shared memory FFT called multiple times, might not work when number of FFT to perform (second executable argument) is too low
MULTIPLE_REUSE does not work
MULTIPLE_REUSE_REGISTERS does not work

params.h  -  doesn't do anything mostly
timer.h - utilities
utils_cuda.h - utilities
utils_file.h - utilities

benchmark.sh  -  will do some basic benchmarking


What is what (implementations):

FFT_CT is Cooley-Tukey FFT DIT
FFT_Pease is Pease FFT DIF
FFT_Stockham is Stockham autosort FFT
FFT_Stockham_4way is Stockham autosort FFT with 4 FFT elements calculated per thread. It is performing better but has higher register usage. Something similar (different algorithm) was used in convolution kernel.
FFT_Stockham_8way is Stockham autosort FFT with 8 FFT elements calculated per thread. It performs even better then 4way but register usage was too high and whole convolution kernel was slower then with 4way.
FFT_Stockham_Radix-R is Stockham autosort FFT which can work with N=R*2^bits FFT sizes. So it is semi-non power-of-two. .cu file contain kernel for arbitrary radix, i.e. N=R^bits, but it is very slow and naive. In params.h there must be #define RADIX defined. For example 3.

Karel Adamek