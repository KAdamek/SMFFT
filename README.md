# SMFFT
This is a shared memory implementation of the fast Fourier transform (FFT) on CUDA GPUs for Astro-Accelerate project.

Compile: 'make' should do that.
You may need to define CUDA_HOME parameter.

## Implementations:
There are two implementations of the FFT algorithm Cooley-Tukey and Stockham FFT algorithm.
### SMFFT_CooleyTukey_C2C
This is a implementation of the Cooley-Tukey FFT algorithm. The code is expected to be called within a GPU kernel but the wrapper used to demonstrate the its functionality can be used. 

The FFT is performed by the CUDA ```__device__``` function   ```do_SMFFT_CT_DIT```, which is expects pointer to a shared memory which contains FFT stored in the float2 format where x is real part and y is the imaginary part. If the result of the FFT transform needs to be reordered into a correct order the size of the shared memory must be as given by appropriate ```FFT_Params``` class instance ```FFT_Params::fft_sm_required```. This might not be always required for example for convolutions.

The CUDA kernel ```SMFFT_DIT_external``` serves as a wrapper for the SMFFT device function. This might be useful if one uses no-reorder variant and need filters which will be applied in the same order as the Fourier transformed data.

The kernels are templated are require class FFT_Params which contains information about what kind of Fourier transform is required, size of the FFT and if user requires correct (ordered output). This class and its variants are contained in ```SM_FFT_parameters.cuh```. 
* ```fft_exp``` is log2(fft_length)
* ```fft_sm_required``` is the required shared memory by the SMFFT and it is calculated as (fft_length/32)*33
* ```fft_direction``` is 0 for forward transform and 1 for inverse transform
* ```fft_reorder``` is 0 for no-reorder 1 for reorder (correctly ordered output)

### SMFFT_Stockham_C2C
Contains GPU implementation of the autosort Stockham FFT algorithm. The algorithm produces correctly ordered output without explicit reordering step. The basic functions and usage is similar to the Cooley-Tukey implementation.

### SMFFT_Stockham_R2C_C2R
Is the same implementation as ```SMFFT_Stockham_C2C``` but extended to handle R2C and C2R Fourier transformation efficiently.

## Files:
```FFT.c```
Host related stuff. Allocated host memory and generate random data. Also performs checks if results are correct.


```FFT-GPU-32bit*.cu```
Device related stuff + kernel. It allocates necessary memory on the device and takes care of transfers HOST <-> DEVICE. Kernel invocation is in function FFT_external_benchmark(...)

FFT_external_benchmark(
```float2 d_input  ``` - complex input (time-domain vectors x=real; y=imaginary)
```float2 d_output ``` - FFT result (frequency-domain vectors x=real; y=imaginary)
```int    FFT_size ``` - FFT size
```int    nFFTs    ``` - number of time-series to transform
```bool   inverse  ``` - transform direction true for inverse Fourier transform
```bool   reorder  ``` - enable reordering true for correctly ordered output
```double *FFT_time``` - execution time
);

FFT_multiple_benchmark(...) runs multiple FFT transforms per CUDA kernel as such simulates the performance of the SMFFT when run from CUDA kernel. It takes same parameters as FFT_external_benchmark.

the kernel itself FFT_GPU_external(...) requires following
```template<class const_params>```
```SMFFT_DIT_external(```
```float2 d_input```  - complex input (time-domain vectors)
```float2 d_output``` - FFT result (frequency-domain vectors)
```);```
this only transfers data to shared memory then calls the FFT device function and then writes data back to global memory.

Function which does the FFT is ```do_SMFFT_CT_DIT```
```do_SMFFT_CT_DIT(```
```float2 s_input``` - shared memory with time-domain vector
```);```
this function perform 'in-place' FFT.


```SM_FFT_parameters.cuh```
Contains definition of the class ```FFT_Params```.


```debug.h```
controls what will code do. 
It is work in progress...
DEBUG activate printing stuff to console

CUFFT enables calculation of the Fourier transform using cuFFT library
EXTERNAL enables calculation of the Fourier transform using shared memory FFT
MULTIPLE enables benchmark which calculates multiples FFT per kernel which removes the limitation of the global memory and better shows performance of the shared memory FFT as would be called from a CUDA kernel 

timer.h - utilities
utils_cuda.h - utilities

## Benchmark
GPU used is V100 32GB with CUDA 10.. Time is in miliseconds. First time is for FFT_multiple benchmark the second time in square brackets is for FFT_external_benchmark which is limited by device memory bandwidth. The input data size is 4GB. The number of FFTs calculated is in square brackets.

FFT size    | Cooley-Tukey | Cooley-Tukey reorder | Stockham     | cuFFT 
--------    | ------------ | -------------------- | ------------ | ----- 
32 [16M]    | 2.04 [10.45] | 2.43 [10.45]         | NA           | NA [10.52]
64 [8M]     | 2.54 [10.45] | 3.93 [10.45]         | NA           | NA [10.45]
128 [4M]    | 3.45 [10.47] | 4.89 [10.47]         | NA           | NA [10.47]
256 [2M]    | 3.95 [10.46] | 5.63 [10.46]         | 6.70 [10.46] | NA [10.55]
512 [1M]    | 4.43 [10.40] | 6.07 [10.40]         | 6.77 [10.39] | NA [10.52]
1024 [524k] | 5.01 [10.41] | 6.16 [10.41]         | 6.90 [10.41] | NA [10.50]
2048 [262k] | 5.77 [10.50] | 7.72 [10.50]         | 7.63 [10.53] | NA [10.49]
4096 [131k] | 6.80 [10.75] | 9.47 [10.75]         | 8.95 [11.52] | NA [10.65]


Karel Adamek 2020-07-200