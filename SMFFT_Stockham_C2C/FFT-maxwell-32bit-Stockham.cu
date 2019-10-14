#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "debug.h"
#include "timer.h"
#include "utils_cuda.h"

#define WARP 32


int device=0;

class FFT_ConstParams {
public:
	static const int fft_exp = -1;
	static const int fft_length = -1;
	static const int fft_half = -1;
	static const int warp = 32;
};

class FFT_256 : public FFT_ConstParams {
	public:
	static const int fft_exp = 8;
	static const int fft_quarter = 64;
	static const int fft_half = 128;
	static const int fft_threequarters = 192;
	static const int fft_length = 256;
	
};

class FFT_512 : public FFT_ConstParams {
	public:
	static const int fft_exp = 9;
	static const int fft_quarter = 128;
	static const int fft_half = 256;
	static const int fft_threequarters = 384;
	static const int fft_length = 512;
};

class FFT_1024 : public FFT_ConstParams {
	public:
	static const int fft_exp = 10;
	static const int fft_quarter = 256;
	static const int fft_half = 512;
	static const int fft_threequarters = 768;
	static const int fft_length = 1024;
};

class FFT_2048 : public FFT_ConstParams {
	public:
	static const int fft_exp = 11;
	static const int fft_quarter = 512;
	static const int fft_half = 1024;
	static const int fft_threequarters = 1536;
	static const int fft_length = 2048;
};

class FFT_4096 : public FFT_ConstParams {
	public:
	static const int fft_exp = 12;
	static const int fft_quarter = 1024;
	static const int fft_half = 2048;
	static const int fft_threequarters = 3072;
	static const int fft_length = 4096;
};


__device__ __inline__ float2 Get_W_value(int N, int m){
	float2 ctemp;
	//ctemp.x=-cosf( 6.283185f*fdividef( (float) m, (float) N ) - 3.141592654f );
	//ctemp.y=sinf( 6.283185f*fdividef( (float) m, (float) N ) - 3.141592654f );
	//ctemp.x=cosf( 2.0f*3.141592654f*fdividef( (float) m, (float) N) );
	//ctemp.y=sinf( 2.0f*3.141592654f*fdividef( (float) m, (float) N) );
	sincosf(6.283185308f*fdividef( (float) m, (float) N), &ctemp.y, &ctemp.x);
	return(ctemp);
}

__device__ __inline__ float shfl(float *value, int par){
	#if (CUDART_VERSION >= 9000)
		return(__shfl_sync(0xffffffff, (*value), par));
	#else
		return(__shfl((*value), par));
	#endif
}

/*
__device__ __inline__ float2 Get_W_value_float(float N, float m){
	float2 ctemp;
	ctemp.x=-cosf( 6.283185f*fdividef( m, N) - 3.141592654f );
	ctemp.y=sinf( 6.283185f*fdividef( m, N) - 3.141592654f );
	return(ctemp);
}
*/

template<class const_params>
__device__ void do_FFT_Stockham_mk6(float2 *s_input){ // in-place
	float2 SA_DFT_value_even, SA_DFT_value_odd;
	float2 SB_DFT_value_even, SB_DFT_value_odd; 
	float2 SA_ftemp2, SA_ftemp;
	float2 SB_ftemp2, SB_ftemp;
	float2 W;
	
	int r, j, k, PoT, PoTm1;

	//-----> FFT
	//--> 

	//int A_index=threadIdx.x;
	//int B_index=threadIdx.x + const_params::fft_half;
	
	PoT=1;
	PoTm1=0;
	//------------------------------------------------------------
	// First iteration
		PoTm1=PoT;
		PoT=PoT<<1;
		
		j=threadIdx.x;
		
		
		SA_ftemp  = s_input[threadIdx.x];
		SA_ftemp2 = s_input[threadIdx.x + const_params::fft_half];
		SA_DFT_value_even.x = SA_ftemp.x + SA_ftemp2.x;
		SA_DFT_value_even.y = SA_ftemp.y + SA_ftemp2.y;
		SA_DFT_value_odd.x  = SA_ftemp.x - SA_ftemp2.x;
		SA_DFT_value_odd.y  = SA_ftemp.y - SA_ftemp2.y;
		
		SB_ftemp  = s_input[threadIdx.x + const_params::fft_quarter];
		SB_ftemp2 = s_input[threadIdx.x + const_params::fft_threequarters];
		SB_DFT_value_even.x = SB_ftemp.x + SB_ftemp2.x;
		SB_DFT_value_even.y = SB_ftemp.y + SB_ftemp2.y;
		SB_DFT_value_odd.x  = SB_ftemp.x - SB_ftemp2.x;
		SB_DFT_value_odd.y  = SB_ftemp.y - SB_ftemp2.y;
		
		__syncthreads();
		s_input[j*PoT]         = SA_DFT_value_even;
		s_input[j*PoT + PoTm1] = SA_DFT_value_odd;
		s_input[j*PoT + const_params::fft_half]         = SB_DFT_value_even;
		s_input[j*PoT + PoTm1 + const_params::fft_half] = SB_DFT_value_odd;
		__syncthreads();
	// First iteration
	//------------------------------------------------------------
	
	for(r=2;r<6;r++){
		PoTm1=PoT;
		PoT=PoT<<1;
		
		j=threadIdx.x>>(r-1);
		k=threadIdx.x & (PoTm1-1);
		
		W=Get_W_value(PoT,k);
		
		SA_ftemp  = s_input[threadIdx.x];
		SA_ftemp2 = s_input[threadIdx.x + const_params::fft_half];
		SA_DFT_value_even.x = SA_ftemp.x + W.x*SA_ftemp2.x - W.y*SA_ftemp2.y;
		SA_DFT_value_even.y = SA_ftemp.y + W.x*SA_ftemp2.y + W.y*SA_ftemp2.x;
		SA_DFT_value_odd.x  = SA_ftemp.x - W.x*SA_ftemp2.x + W.y*SA_ftemp2.y;
		SA_DFT_value_odd.y  = SA_ftemp.y - W.x*SA_ftemp2.y - W.y*SA_ftemp2.x;
		
		SB_ftemp  = s_input[threadIdx.x + const_params::fft_quarter];
		SB_ftemp2 = s_input[threadIdx.x + const_params::fft_threequarters];
		SB_DFT_value_even.x = SB_ftemp.x + W.x*SB_ftemp2.x - W.y*SB_ftemp2.y;
		SB_DFT_value_even.y = SB_ftemp.y + W.x*SB_ftemp2.y + W.y*SB_ftemp2.x;
		SB_DFT_value_odd.x  = SB_ftemp.x - W.x*SB_ftemp2.x + W.y*SB_ftemp2.y;
		SB_DFT_value_odd.y  = SB_ftemp.y - W.x*SB_ftemp2.y - W.y*SB_ftemp2.x;
		
		__syncthreads();
		s_input[j*PoT + k]         = SA_DFT_value_even;
		s_input[j*PoT + k + PoTm1] = SA_DFT_value_odd;
		s_input[j*PoT + k + const_params::fft_half]         = SB_DFT_value_even;
		s_input[j*PoT + k + PoTm1 + const_params::fft_half] = SB_DFT_value_odd;
		__syncthreads();
	}
	
	
	for(r=6;r<=const_params::fft_exp-1;r++){
		PoTm1=PoT;
		PoT=PoT<<1;
		
		j=threadIdx.x>>(r-1);
		k=threadIdx.x & (PoTm1-1);
		
		W=Get_W_value(PoT,k);
		
		SA_ftemp  = s_input[threadIdx.x];
		SA_ftemp2 = s_input[threadIdx.x + const_params::fft_half];
		SA_DFT_value_even.x = SA_ftemp.x + W.x*SA_ftemp2.x - W.y*SA_ftemp2.y;
		SA_DFT_value_even.y = SA_ftemp.y + W.x*SA_ftemp2.y + W.y*SA_ftemp2.x;
		SA_DFT_value_odd.x  = SA_ftemp.x - W.x*SA_ftemp2.x + W.y*SA_ftemp2.y;
		SA_DFT_value_odd.y  = SA_ftemp.y - W.x*SA_ftemp2.y - W.y*SA_ftemp2.x;
		
		SB_ftemp  = s_input[threadIdx.x + const_params::fft_quarter];
		SB_ftemp2 = s_input[threadIdx.x + const_params::fft_threequarters];
		SB_DFT_value_even.x = SB_ftemp.x + W.x*SB_ftemp2.x - W.y*SB_ftemp2.y;
		SB_DFT_value_even.y = SB_ftemp.y + W.x*SB_ftemp2.y + W.y*SB_ftemp2.x;
		SB_DFT_value_odd.x  = SB_ftemp.x - W.x*SB_ftemp2.x + W.y*SB_ftemp2.y;
		SB_DFT_value_odd.y  = SB_ftemp.y - W.x*SB_ftemp2.y - W.y*SB_ftemp2.x;
		
		__syncthreads();
		s_input[j*PoT + k]         = SA_DFT_value_even;
		s_input[j*PoT + k + PoTm1] = SA_DFT_value_odd;
		s_input[j*PoT + k + const_params::fft_half]         = SB_DFT_value_even;
		s_input[j*PoT + k + PoTm1 + const_params::fft_half] = SB_DFT_value_odd;
		__syncthreads();
	}
	// Last iteration
	{
		j = 0;
		k = threadIdx.x;
		
		float2 WA = Get_W_value(const_params::fft_length, threadIdx.x);
		SA_ftemp  = s_input[threadIdx.x];
		SA_ftemp2 = s_input[threadIdx.x + const_params::fft_half];
		SA_DFT_value_even.x = SA_ftemp.x + WA.x*SA_ftemp2.x - WA.y*SA_ftemp2.y;
		SA_DFT_value_even.y = SA_ftemp.y + WA.x*SA_ftemp2.y + WA.y*SA_ftemp2.x;
		SA_DFT_value_odd.x  = SA_ftemp.x - WA.x*SA_ftemp2.x + WA.y*SA_ftemp2.y;
		SA_DFT_value_odd.y  = SA_ftemp.y - WA.x*SA_ftemp2.y - WA.y*SA_ftemp2.x;
		
		float2 WB = Get_W_value(const_params::fft_length, threadIdx.x + const_params::fft_quarter);
		SB_ftemp  = s_input[threadIdx.x + const_params::fft_quarter];
		SB_ftemp2 = s_input[threadIdx.x + const_params::fft_threequarters];
		SB_DFT_value_even.x = SB_ftemp.x + WB.x*SB_ftemp2.x - WB.y*SB_ftemp2.y;
		SB_DFT_value_even.y = SB_ftemp.y + WB.x*SB_ftemp2.y + WB.y*SB_ftemp2.x;
		SB_DFT_value_odd.x  = SB_ftemp.x - WB.x*SB_ftemp2.x + WB.y*SB_ftemp2.y;
		SB_DFT_value_odd.y  = SB_ftemp.y - WB.x*SB_ftemp2.y - WB.y*SB_ftemp2.x;
		
		__syncthreads();
		s_input[threadIdx.x]                          = SA_DFT_value_even;
		s_input[threadIdx.x + const_params::fft_half] = SA_DFT_value_odd;
		s_input[threadIdx.x + const_params::fft_quarter]       = SB_DFT_value_even;
		s_input[threadIdx.x + const_params::fft_threequarters] = SB_DFT_value_odd;
		__syncthreads();
	}
	
	//-------> END
	
	__syncthreads();
}


template<class const_params>
__global__ void FFT_GPU_external(float2 *d_input, float2* d_output) {
	extern __shared__ float2 s_input[];
	s_input[threadIdx.x]                                   = d_input[threadIdx.x + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_quarter]       = d_input[threadIdx.x + const_params::fft_quarter + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_half]          = d_input[threadIdx.x + const_params::fft_half + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_threequarters] = d_input[threadIdx.x + const_params::fft_threequarters + blockIdx.x*const_params::fft_length];
	__syncthreads();
	
	do_FFT_Stockham_mk6<const_params>(s_input);
	
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length]                                   = s_input[threadIdx.x];
	d_output[threadIdx.x + const_params::fft_quarter + blockIdx.x*const_params::fft_length]       = s_input[threadIdx.x + const_params::fft_quarter];
	d_output[threadIdx.x + const_params::fft_half + blockIdx.x*const_params::fft_length]          = s_input[threadIdx.x + const_params::fft_half];
	d_output[threadIdx.x + const_params::fft_threequarters + blockIdx.x*const_params::fft_length] = s_input[threadIdx.x + const_params::fft_threequarters];
}

template<class const_params>
__global__ void FFT_GPU_multiple(float2 *d_input, float2* d_output) {
	extern __shared__ float2 s_input[];
	s_input[threadIdx.x]                                   = d_input[threadIdx.x + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_quarter]       = d_input[threadIdx.x + const_params::fft_quarter + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_half]          = d_input[threadIdx.x + const_params::fft_half + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_threequarters] = d_input[threadIdx.x + const_params::fft_threequarters + blockIdx.x*const_params::fft_length];
	__syncthreads();
	

	for(int f=0;f<100;f++){
		do_FFT_Stockham_mk6<const_params>(s_input);
	}
	
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length]                                   = s_input[threadIdx.x];
	d_output[threadIdx.x + const_params::fft_quarter + blockIdx.x*const_params::fft_length]       = s_input[threadIdx.x + const_params::fft_quarter];
	d_output[threadIdx.x + const_params::fft_half + blockIdx.x*const_params::fft_length]          = s_input[threadIdx.x + const_params::fft_half];
	d_output[threadIdx.x + const_params::fft_threequarters + blockIdx.x*const_params::fft_length] = s_input[threadIdx.x + const_params::fft_threequarters];
}


int Max_columns_in_memory_shared(int FFT_size, int nFFTs) {
	long int nColumns,maxgrid_x;

	size_t free_mem,total_mem;
	cudaDeviceProp devProp;
	
	checkCudaErrors(cudaSetDevice(device));
	checkCudaErrors(cudaGetDeviceProperties(&devProp,device));
	maxgrid_x = devProp.maxGridSize[0];
	cudaMemGetInfo(&free_mem,&total_mem);
	
	nColumns=((long int) free_mem)/(2.0*sizeof(float2)*FFT_size);
	if(nColumns>maxgrid_x) nColumns=maxgrid_x;
	nColumns=(int) nColumns*0.9;
	return(nColumns);
}


void FFT_init(){
	//---------> Specific nVidia stuff
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
}


void FFT_external_benchmark(float2 *d_input, float2 *d_output, int FFT_size, int nFFTs, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	int nCUDAblocks_x=nFFTs;
	int nCUDAblocks_y=1;
	
	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);
	dim3 blockSize(FFT_size/4, 1, 1);
	
	//---------> FFT part
	timer.Start();
	switch(FFT_size) {
		case 256:
			FFT_GPU_external<FFT_256><<<gridSize, blockSize, FFT_size*8>>>(d_input, d_output);
			break;
			
		case 512:
			FFT_GPU_external<FFT_512><<<gridSize, blockSize, FFT_size*8>>>(d_input, d_output);
			break;
		
		case 1024:
			FFT_GPU_external<FFT_1024><<<gridSize, blockSize, FFT_size*8>>>(d_input, d_output);
			break;

		case 2048:
			FFT_GPU_external<FFT_2048><<<gridSize, blockSize, FFT_size*8>>>(d_input, d_output);
			break;
			
		case 4096:
			FFT_GPU_external<FFT_4096><<<gridSize, blockSize, FFT_size*8>>>(d_input, d_output);
			break;
		
		default : 
			printf("Error wrong FFT length!\n");
			break;
	}
	timer.Stop();
	
	*FFT_time += timer.Elapsed();
}


void FFT_multiple_benchmark(float2 *d_input, float2 *d_output, int FFT_size, int nFFTs, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize_multiple((int) (nFFTs/100), 1, 1);
	dim3 blockSize(FFT_size/4, 1, 1);
	
	//---------> FIR filter part
	timer.Start();
	switch(FFT_size) {
		case 256:
			FFT_GPU_multiple<FFT_256><<<gridSize_multiple, blockSize, FFT_size*8>>>(d_input, d_output);
			break;
			
		case 512:
			FFT_GPU_multiple<FFT_512><<<gridSize_multiple, blockSize, FFT_size*8>>>(d_input, d_output);
			break;
		
		case 1024:
			FFT_GPU_multiple<FFT_1024><<<gridSize_multiple, blockSize, FFT_size*8>>>(d_input, d_output);
			break;

		case 2048:
			FFT_GPU_multiple<FFT_2048><<<gridSize_multiple, blockSize, FFT_size*8>>>(d_input, d_output);
			break;
			
		case 4096:
			FFT_GPU_multiple<FFT_4096><<<gridSize_multiple, blockSize, FFT_size*8>>>(d_input, d_output);
			break;
		
		default :
			printf("Error wrong FFT length!\n");
			break;
	}
	timer.Stop();
	
	*FFT_time += timer.Elapsed();
}



// ***********************************************************************************
int GPU_cuFFT(float2 *h_input, float2 *h_output, int FFT_size, int nFFTs, int nRuns, double *single_ex_time){
	//---------> Initial nVidia stuff
	int devCount;
	size_t free_mem,total_mem;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if(devCount>device) checkCudaErrors(cudaSetDevice(device));
	
	//---------> Checking memory
	cudaMemGetInfo(&free_mem,&total_mem);
	if(DEBUG) printf("\n  Device has %0.3f MB of total memory, which %0.3f MB is available.\n", ((float) total_mem)/(1024.0*1024.0), (float) free_mem/(1024.0*1024.0));
	size_t input_size = FFT_size*nFFTs;
	size_t output_size = FFT_size*nFFTs;
	size_t total_memory_required_bytes = input_size*sizeof(float2) + output_size*sizeof(float2);
	if(total_memory_required_bytes>free_mem) {
		printf("Error: Not enough memory! Input data are too big for the device.\n");
		return(1);
	}
	
	//----------> Memory allocation
	float2 *d_input;
	float2 *d_output;
	checkCudaErrors(cudaMalloc((void **) &d_input,  sizeof(float2)*input_size));
	checkCudaErrors(cudaMalloc((void **) &d_output, sizeof(float2)*output_size));
	
	checkCudaErrors(cudaMemcpy(d_input, h_input, input_size*sizeof(float2), cudaMemcpyHostToDevice));
	
	//---------> Measurements
	double time_cuFFT = 0;
	GpuTimer timer;
		
	//--------------------------------------------------
	//-------------------------> cuFFT
	cufftHandle plan;
	cufftResult error;
	error = cufftPlan1d(&plan, FFT_size, CUFFT_C2C, nFFTs);
	if (CUFFT_SUCCESS != error){
		printf("CUFFT error: %d", error);
	}
	
	timer.Start();
	cufftExecC2C(plan, (cufftComplex *)d_input, (cufftComplex *)d_output, CUFFT_INVERSE);
	timer.Stop();
	time_cuFFT += timer.Elapsed();
	
	checkCudaErrors(cudaMemcpy( h_output, d_output, output_size*sizeof(float2), cudaMemcpyDeviceToHost));
	
	cufftDestroy(plan);
	//-----------------------------------<
	//--------------------------------------------------
	
	printf("  FFT size: %d; cuFFT time = %0.3f ms;\n", FFT_size, time_cuFFT);
	
	cudaDeviceSynchronize();
	
	//---------> Copy Device -> Host
	checkCudaErrors(cudaMemcpy(h_output, d_output, output_size*sizeof(float2), cudaMemcpyDeviceToHost));

	//---------> error check -----
	checkCudaErrors(cudaGetLastError());
	
	//---------> Feeing allocated resources
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_output));
	
	return(0);
}


int GPU_FFT_C2C_Stockham(float2 *h_input, float2 *h_smFFT_output, int FFT_size, int nFFTs, int nRuns, double *single_ex_time, double *multi_ex_time){
	//---------> Initial nVidia stuff
	int devCount;
	size_t free_mem,total_mem;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if(devCount>device) checkCudaErrors(cudaSetDevice(device));
	
	//---------> Checking memory
	cudaMemGetInfo(&free_mem,&total_mem);
	if(DEBUG) printf("\n  Device has %0.3f MB of total memory, which %0.3f MB is available.\n", ((float) total_mem)/(1024.0*1024.0), (float) free_mem/(1024.0*1024.0));
	size_t input_size = FFT_size*nFFTs;
	size_t output_size = FFT_size*nFFTs;
	size_t input_size_bytes  = FFT_size*nFFTs*sizeof(float2);
	size_t output_size_bytes = FFT_size*nFFTs*sizeof(float2);
	size_t total_memory_required_bytes = input_size*sizeof(float2) + output_size*sizeof(float2);
	if(total_memory_required_bytes>free_mem) {
		printf("Error: Not enough memory! Input data is too big for the device.\n");
		return(1);
	}
	
	//---------> Measurements
	double time_FFT_external = 0, time_FFT_multiple = 0;
	GpuTimer timer; 
	
	//---------> Memory allocation
	float2 *d_output;
	float2 *d_input;
	timer.Start();
	checkCudaErrors(cudaMalloc((void **) &d_input,  input_size_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_output, output_size_bytes));
	timer.Stop();

	
	if(MULTIPLE){
		if (DEBUG) printf("  Running shared memory FFT 100 (Stockham) times per GPU kernel (eliminates device memory)... ");
		FFT_init();
		double total_time_FFT_multiple = 0;
		for(int f=0; f<nRuns; f++){
			checkCudaErrors(cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice));
			FFT_multiple_benchmark(d_input, d_output, FFT_size, nFFTs, &total_time_FFT_multiple);
		}
		time_FFT_multiple = total_time_FFT_multiple/nRuns;
		if (DEBUG) printf("done in %g ms.\n", time_FFT_multiple);
		*multi_ex_time = time_FFT_multiple;
	}
    
	if(EXTERNAL){
		if (DEBUG) printf("  Running shared memory FFT (Stockham)... ");
		FFT_init();
		double total_time_FFT_external = 0;
		for(int f=0; f<nRuns; f++){
			checkCudaErrors(cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice));
			FFT_external_benchmark(d_input, d_output, FFT_size, nFFTs, &total_time_FFT_external);
		}
		time_FFT_external = total_time_FFT_external/nRuns;
		if (DEBUG) printf("done in %g ms.\n", time_FFT_external);
		*single_ex_time = time_FFT_external;
	}
	
		
	//-----> Copy chunk of output data to host
	checkCudaErrors(cudaMemcpy( h_smFFT_output, d_output, output_size_bytes, cudaMemcpyDeviceToHost));

	//---------> error check -----
	checkCudaErrors(cudaGetLastError());
	
	//---------> Feeing allocated resources
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_output));
	
	printf("  SH FFT normal = %0.3f ms; SM FFT multiple times = %0.3f ms\n", time_FFT_external, time_FFT_multiple);
	
	return(0);
}
