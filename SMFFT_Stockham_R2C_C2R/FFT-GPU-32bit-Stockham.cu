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

class FFT_ConstDirection {
public:
	static const int fft_direction = -1;
};

class FFT_forward : public FFT_ConstDirection {
public:
	static const int fft_direction = 0;
};

class FFT_inverse : public FFT_ConstDirection {
public:
	static const int fft_direction = 1;
};



__device__ __inline__ float2 Get_W_value(int N, int m){
	float2 ctemp;
	sincosf(-6.283185308f*fdividef( (float) m, (float) N), &ctemp.y, &ctemp.x);
	return(ctemp);
}


__device__ __inline__ float2 Get_W_value_inverse(int N, int m){
	float2 ctemp;
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

template<class const_params, class const_direction>
__device__ void do_FFT_Stockham_C2C(float2 *s_input){ // in-place
	float2 SA_DFT_value_even, SA_DFT_value_odd;
	float2 SB_DFT_value_even, SB_DFT_value_odd; 
	float2 SA_ftemp2, SA_ftemp;
	float2 SB_ftemp2, SB_ftemp;
	float2 W;
	
	int r, j, k, PoT, PoTm1;

	//-----> FFT
	//--> 
	
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
		
		if(const_direction::fft_direction==0) {
			W = Get_W_value(PoT,k);
		}
		else {
			W = Get_W_value_inverse(PoT,k);
		}
		
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
		
		if(const_direction::fft_direction==0) {
			W = Get_W_value(PoT,k);
		}
		else {
			W = Get_W_value_inverse(PoT,k);
		}
		
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
		
		float2 WA;
		if(const_direction::fft_direction==0) {
			WA = Get_W_value(const_params::fft_length, threadIdx.x);
		}
		else {
			WA = Get_W_value_inverse(const_params::fft_length, threadIdx.x);
		}
		SA_ftemp  = s_input[threadIdx.x];
		SA_ftemp2 = s_input[threadIdx.x + const_params::fft_half];
		SA_DFT_value_even.x = SA_ftemp.x + WA.x*SA_ftemp2.x - WA.y*SA_ftemp2.y;
		SA_DFT_value_even.y = SA_ftemp.y + WA.x*SA_ftemp2.y + WA.y*SA_ftemp2.x;
		SA_DFT_value_odd.x  = SA_ftemp.x - WA.x*SA_ftemp2.x + WA.y*SA_ftemp2.y;
		SA_DFT_value_odd.y  = SA_ftemp.y - WA.x*SA_ftemp2.y - WA.y*SA_ftemp2.x;
		
		float2 WB;
		if(const_direction::fft_direction==0) {
			WB = Get_W_value(const_params::fft_length, threadIdx.x + const_params::fft_quarter);
		}
		else {
			WB = Get_W_value_inverse(const_params::fft_length, threadIdx.x + const_params::fft_quarter);
		}
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


template<class const_params, class const_direction>
__device__ void do_FFT_Stockham_R2C_C2R(float2 *s_input){
	float2 one_half;
	if(const_direction::fft_direction==0) {
		one_half.x = 0.5f;
		one_half.y = -0.5f;
		do_FFT_Stockham_C2C<const_params,FFT_forward>(s_input);
	}
	else {
		one_half.x = -0.5f;
		one_half.y = 0.5f;
		if(threadIdx.x==0) {
			float2 L, F;
			L = s_input[0];
			F.x = 0.5f*(L.x + L.y);
			F.y = 0.5f*(L.x - L.y); 
			s_input[0] = F;			
		}
	}

	float2 SA_A, SA_B, SA_W, SA_H1, SA_H2, SA_F1, SA_F2;
	SA_A = s_input[threadIdx.x + 1];
	SA_B = s_input[const_params::fft_length - threadIdx.x - 1];
	SA_H1.x =       0.5f*(SA_A.x + SA_B.x);
	SA_H1.y =       0.5f*(SA_A.y - SA_B.y);
	SA_H2.x = one_half.x*(SA_A.y + SA_B.y);
	SA_H2.y = one_half.y*(SA_A.x - SA_B.x);
	if(const_direction::fft_direction==0) {
		SA_W = Get_W_value(const_params::fft_length*2, threadIdx.x + 1);
	}
	else {
		SA_W = Get_W_value_inverse(const_params::fft_length*2, threadIdx.x + 1);
	}
	SA_F1.x =  SA_H1.x + SA_W.x*SA_H2.x - SA_W.y*SA_H2.y;
	SA_F1.y =  SA_H1.y + SA_W.x*SA_H2.y + SA_W.y*SA_H2.x;
	SA_F2.x =  SA_H1.x - SA_W.x*SA_H2.x + SA_W.y*SA_H2.y;
	SA_F2.y = -SA_H1.y + SA_W.x*SA_H2.y + SA_W.y*SA_H2.x;
	s_input[threadIdx.x + 1] = SA_F1;
	s_input[const_params::fft_length - threadIdx.x - 1] = SA_F2;
	

	float2 SB_A, SB_B, SB_W, SB_H1, SB_H2, SB_F1, SB_F2;
	SB_A = s_input[threadIdx.x + 1 + const_params::fft_quarter];
	SB_B = s_input[const_params::fft_length - threadIdx.x - 1 - const_params::fft_quarter];
	SB_H1.x =       0.5f*(SB_A.x + SB_B.x);
	SB_H1.y =       0.5f*(SB_A.y - SB_B.y);
	SB_H2.x = one_half.x*(SB_A.y + SB_B.y);
	SB_H2.y = one_half.y*(SB_A.x - SB_B.x);
	if(const_direction::fft_direction==0) {
		SB_W = Get_W_value(const_params::fft_length*2, threadIdx.x + 1 + const_params::fft_quarter);
	}
	else {
		SB_W = Get_W_value_inverse(const_params::fft_length*2, threadIdx.x + 1 + const_params::fft_quarter);
	}
	SB_F1.x =  SB_H1.x + SB_W.x*SB_H2.x - SB_W.y*SB_H2.y;
	SB_F1.y =  SB_H1.y + SB_W.x*SB_H2.y + SB_W.y*SB_H2.x;
	SB_F2.x =  SB_H1.x - SB_W.x*SB_H2.x + SB_W.y*SB_H2.y;
	SB_F2.y = -SB_H1.y + SB_W.x*SB_H2.y + SB_W.y*SB_H2.x;
	s_input[threadIdx.x + 1 + const_params::fft_quarter] = SB_F1;
	s_input[const_params::fft_length - threadIdx.x - 1 - const_params::fft_quarter] = SB_F2;

	__syncthreads();
	
	if(const_direction::fft_direction==0) {
		if(threadIdx.x==0) {
			float2 L, F;
			L = s_input[0];
			F.x = L.x + L.y;
			F.y = L.x - L.y;
			s_input[0] = F;
		}
	}
	else {
		do_FFT_Stockham_C2C<const_params,FFT_inverse>(s_input);
	}
}




template<class const_params, class const_direction>
__global__ void FFT_GPU_R2C_C2R_external(float2 *d_input, float2* d_output) {
	__shared__ float2 s_input[const_params::fft_length + 1];
	s_input[threadIdx.x]                                   = d_input[threadIdx.x + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_quarter]       = d_input[threadIdx.x + const_params::fft_quarter + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_half]          = d_input[threadIdx.x + const_params::fft_half + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_threequarters] = d_input[threadIdx.x + const_params::fft_threequarters + blockIdx.x*const_params::fft_length];
	__syncthreads();
	
	do_FFT_Stockham_R2C_C2R<const_params, const_direction>(s_input);

	__syncthreads();
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length]                                   = s_input[threadIdx.x];
	d_output[threadIdx.x + const_params::fft_quarter + blockIdx.x*const_params::fft_length]       = s_input[threadIdx.x + const_params::fft_quarter];
	d_output[threadIdx.x + const_params::fft_half + blockIdx.x*const_params::fft_length]          = s_input[threadIdx.x + const_params::fft_half];
	d_output[threadIdx.x + const_params::fft_threequarters + blockIdx.x*const_params::fft_length] = s_input[threadIdx.x + const_params::fft_threequarters];
}

template<class const_params, class const_direction>
__global__ void FFT_GPU_R2C_C2R_multiple(float2 *d_input, float2* d_output) {
	__shared__ float2 s_input[const_params::fft_length + 1];
	s_input[threadIdx.x]                                   = d_input[threadIdx.x + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_quarter]       = d_input[threadIdx.x + const_params::fft_quarter + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_half]          = d_input[threadIdx.x + const_params::fft_half + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_threequarters] = d_input[threadIdx.x + const_params::fft_threequarters + blockIdx.x*const_params::fft_length];
	__syncthreads();
	
	for(int f=0;f<100;f++){
		do_FFT_Stockham_R2C_C2R<const_params, const_direction>(s_input);
	}
	
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length]                                   = s_input[threadIdx.x];
	d_output[threadIdx.x + const_params::fft_quarter + blockIdx.x*const_params::fft_length]       = s_input[threadIdx.x + const_params::fft_quarter];
	d_output[threadIdx.x + const_params::fft_half + blockIdx.x*const_params::fft_length]          = s_input[threadIdx.x + const_params::fft_half];
	d_output[threadIdx.x + const_params::fft_threequarters + blockIdx.x*const_params::fft_length] = s_input[threadIdx.x + const_params::fft_threequarters];
}



void FFT_init(){
	//---------> Specific nVidia stuff
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
}



void FFT_external_benchmark(float *d_input, float *d_output, int FFT_size, int nFFTs, int inverse, double *FFT_time){
	GpuTimer timer;
	
	dim3 gridSize(nFFTs, 1, 1);
	dim3 blockSize((FFT_size>>1)/4, 1, 1);
	
	//---------> FFT part
	timer.Start();
	switch(FFT_size) {
		case 512:
			if(inverse==0) FFT_GPU_R2C_C2R_external<FFT_256,FFT_forward><<<gridSize, blockSize>>>((float2 *) d_input, (float2 *) d_output);
			else FFT_GPU_R2C_C2R_external<FFT_256,FFT_inverse><<<gridSize, blockSize>>>((float2 *) d_input, (float2 *) d_output);
			break;
		
		case 1024:
			if(inverse==0) FFT_GPU_R2C_C2R_external<FFT_512,FFT_forward><<<gridSize, blockSize>>>((float2 *) d_input, (float2 *) d_output);
			else FFT_GPU_R2C_C2R_external<FFT_512,FFT_inverse><<<gridSize, blockSize>>>((float2 *) d_input, (float2 *) d_output);
			break;

		case 2048:
			if(inverse==0) FFT_GPU_R2C_C2R_external<FFT_1024,FFT_forward><<<gridSize, blockSize>>>((float2 *) d_input, (float2 *) d_output);
			else FFT_GPU_R2C_C2R_external<FFT_1024,FFT_inverse><<<gridSize, blockSize>>>((float2 *) d_input, (float2 *) d_output);
			break;
			
		case 4096:
			if(inverse==0) FFT_GPU_R2C_C2R_external<FFT_2048,FFT_forward><<<gridSize, blockSize>>>((float2 *) d_input, (float2 *) d_output);
			else FFT_GPU_R2C_C2R_external<FFT_2048,FFT_inverse><<<gridSize, blockSize>>>((float2 *) d_input, (float2 *) d_output);
			break;
		
		default :
			printf("Error wrong FFT length!\n");
			break;
	}
	timer.Stop();
	
	*FFT_time += timer.Elapsed();
}


void FFT_multiple_benchmark(float *d_input, float *d_output, int FFT_size, int nFFTs, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize_multiple((int) (nFFTs/100), 1, 1);
	dim3 blockSize((FFT_size>>1)/4, 1, 1);
	
	//---------> FIR filter part
	timer.Start();
	switch(FFT_size) {
		case 512:
			FFT_GPU_R2C_C2R_multiple<FFT_256,FFT_forward><<<gridSize_multiple, blockSize, ((FFT_size>>1)+1)*8>>>((float2 *)d_input, (float2 *)d_output);
			break;
		
		case 1024:
			FFT_GPU_R2C_C2R_multiple<FFT_512,FFT_forward><<<gridSize_multiple, blockSize, ((FFT_size>>1)+1)*8>>>((float2 *)d_input, (float2 *)d_output);
			break;

		case 2048:
			FFT_GPU_R2C_C2R_multiple<FFT_1024,FFT_forward><<<gridSize_multiple, blockSize, ((FFT_size>>1)+1)*8>>>((float2 *)d_input, (float2 *)d_output);
			break;
			
		case 4096:
			FFT_GPU_R2C_C2R_multiple<FFT_2048,FFT_forward><<<gridSize_multiple, blockSize, ((FFT_size>>1)+1)*8>>>((float2 *)d_input, (float2 *)d_output);
			break;
		
		default : 
			printf("Error wrong FFT length!\n");
			break;
	}
	timer.Stop();
	
	*FFT_time += timer.Elapsed();
}



int GPU_cuFFT_R2C(float2 *h_output, float *h_input, int FFT_size, int nFFTs, int nRuns){
	checkCudaErrors(cudaSetDevice(device));
	size_t free_memory, total_memory;
	cudaMemGetInfo(&free_memory,&total_memory);
	double FFT_time = 0;
	GpuTimer timer;
	
	size_t input_size_bytes  = FFT_size*nFFTs*sizeof(float);
	size_t output_size_bytes = ((FFT_size>>1)+1)*nFFTs*sizeof(float2);
	if((input_size_bytes + output_size_bytes) > free_memory) {
		printf("Error not enough free memory!\n");
		return(1);
	}
	
	float *d_input;
	float2 *d_output;
	checkCudaErrors(cudaMalloc((void **) &d_input,  input_size_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_output, output_size_bytes));
	
	checkCudaErrors(cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice));
	
	//---------> FFT
	if(CUFFT){
		cufftHandle plan;
		cufftResult error;
		error = cufftPlan1d(&plan, FFT_size, CUFFT_R2C, nFFTs);
		if (CUFFT_SUCCESS != error){
			printf("CUFFT error: %d", error);
		}
		
		timer.Start();
		for(int r=0; r<nRuns; r++){
			cufftExecR2C(plan, (cufftReal *) d_input, (cufftComplex *)d_output);
		}
		timer.Stop();
		FFT_time = timer.Elapsed();
		
		cufftDestroy(plan);
	}
	
	printf("  cuFFT R2C time: %0.3f ms\n", FFT_time/nRuns);
	
	checkCudaErrors(cudaMemcpy( h_output, d_output, output_size_bytes, cudaMemcpyDeviceToHost));
	
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_output));
	return(0);
}

int GPU_cuFFT_C2R(float *h_output, float2 *h_input, int FFT_size, int nFFTs, int nRuns){
	checkCudaErrors(cudaSetDevice(device));
	size_t free_memory, total_memory;
	cudaMemGetInfo(&free_memory,&total_memory);
	double FFT_time = 0;
	GpuTimer timer;
	
	size_t input_size_bytes  = ((FFT_size>>1)+1)*nFFTs*sizeof(float2);
	size_t output_size_bytes = FFT_size*nFFTs*sizeof(float);
	if((input_size_bytes + output_size_bytes) > free_memory) {
		printf("Error not enough free memory!\n");
		return(1);
	}
	
	float2 *d_input;
	float *d_output;
	checkCudaErrors(cudaMalloc((void **) &d_input,  input_size_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_output, output_size_bytes));
	
	checkCudaErrors(cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice));
	
	//---------> FFT
	if(CUFFT){
		cufftHandle plan;
		cufftResult error;
		error = cufftPlan1d(&plan, FFT_size, CUFFT_C2R, nFFTs);
		if (CUFFT_SUCCESS != error){
			printf("CUFFT error: %d", error);
		}
		
		timer.Start();
		for(int r=0; r<nRuns; r++){
			cufftExecC2R(plan, (cufftComplex *) d_input, (cufftReal *)d_output);
		}
		timer.Stop();
		FFT_time = timer.Elapsed();
		
		cufftDestroy(plan);
	}
	
	printf("  cuFFT C2R time: %0.3f ms\n", FFT_time/nRuns);
	
	checkCudaErrors(cudaMemcpy( h_output, d_output, output_size_bytes, cudaMemcpyDeviceToHost));
	
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_output));
	return(0);
}




int GPU_smFFT_R2C(float2 *h_output, float *h_input, int FFT_size, int nFFTs, int nRuns){
	checkCudaErrors(cudaSetDevice(device));
	size_t free_memory, total_memory;
	cudaMemGetInfo(&free_memory,&total_memory);
	double FFT_external_time = 0;
	double FFT_multiple_time = 0;
	GpuTimer timer;
	
	size_t input_size_bytes  = FFT_size*nFFTs*sizeof(float);
	size_t output_size_bytes = (FFT_size>>1)*nFFTs*sizeof(float2);
	if((input_size_bytes + output_size_bytes) > free_memory) {
		printf("Error not enough free memory!\n");
		return(1);
	}
	
	float *d_input;
	float2 *d_output;
	checkCudaErrors(cudaMalloc((void **) &d_input,  input_size_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_output, output_size_bytes));
	
	checkCudaErrors(cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice));
	
	
		#define TEST_C2R false
		if(TEST_C2R){
			float *d_output_2;
			checkCudaErrors(cudaMalloc((void **) &d_output_2, output_size_bytes));
			
			FFT_init();
			checkCudaErrors(cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice));
			FFT_external_benchmark((float *) d_input, (float *) d_output, FFT_size, nFFTs, 0, &FFT_external_time);
			checkCudaErrors(cudaGetLastError());
			FFT_external_benchmark((float *) d_output, (float *) d_output_2, FFT_size, nFFTs, 1, &FFT_external_time);
			
			float *temp_output;
			temp_output = new float[FFT_size*nFFTs];
			checkCudaErrors(cudaMemcpy( temp_output, d_output_2, FFT_size*nFFTs*sizeof(float), cudaMemcpyDeviceToHost));
			
			float total_c2r_error = 0;
			float c2r_error;
			for(int f=0; f<FFT_size; f++) {
				c2r_error = (h_input[f] - temp_output[f]/(FFT_size>>1))*(h_input[f] - temp_output[f]/(FFT_size>>1));
				if(c2r_error>0) printf("f=%d; error=%f; input:%f; output:%f; ratio:%f\n", f, c2r_error, h_input[f], temp_output[f]/(FFT_size>>1), (temp_output[f]/(FFT_size>>1))/h_input[f]);
				total_c2r_error = total_c2r_error + c2r_error;
			}
			printf("Total error for C2R is %f %f\n", total_c2r_error, total_c2r_error/FFT_size);
			
			delete [] temp_output;
			
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaFree(d_output_2));
		}
	
	
	//---------> FFT
	if(MULTIPLE){
		for(int r=0; r<nRuns; r++){
			FFT_init();
			FFT_multiple_benchmark((float *) d_input, (float *) d_output, FFT_size, nFFTs, &FFT_multiple_time);
		}
	}
	
	cudaMemset(d_output, 0, output_size_bytes);

	if(EXTERNAL){
		for(int r=0; r<nRuns; r++){
			FFT_init();
			FFT_external_benchmark((float *) d_input, (float *) d_output, FFT_size, nFFTs, 0, &FFT_external_time);
		}
	}
	
	printf("  smFFT R2C time: ex: %0.3f ms; mul: %0.3f ms\n", FFT_external_time/nRuns, FFT_multiple_time/nRuns);
	
	checkCudaErrors(cudaMemcpy( h_output, d_output, output_size_bytes, cudaMemcpyDeviceToHost));
	
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_output));
	return(0);
}

int GPU_smFFT_C2R(float *h_output, float2 *h_input, int FFT_size, int nFFTs, int nRuns){
	checkCudaErrors(cudaSetDevice(device));
	size_t free_memory, total_memory;
	cudaMemGetInfo(&free_memory,&total_memory);
	double FFT_external_time = 0;
	GpuTimer timer;
	
	size_t input_size_bytes  = (FFT_size>>1)*nFFTs*sizeof(float2);
	size_t output_size_bytes = FFT_size*nFFTs*sizeof(float);
	if((input_size_bytes + output_size_bytes) > free_memory) {
		printf("Error not enough free memory!\n");
		return(1);
	}
	
	float2 *d_input;
	float *d_output;
	checkCudaErrors(cudaMalloc((void **) &d_input,  input_size_bytes));
	checkCudaErrors(cudaMalloc((void **) &d_output, output_size_bytes));
	
	checkCudaErrors(cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice));
	
	//---------> FFT
	if(EXTERNAL){
		for(int r=0; r<nRuns; r++){
			FFT_init();
			FFT_external_benchmark((float *)d_input, (float *) d_output, FFT_size, nFFTs, 1, &FFT_external_time);
		}
	}
	
	printf("  smFFT C2R time: ex: %0.3f ms;\n", FFT_external_time/nRuns);
	
	checkCudaErrors(cudaMemcpy( h_output, d_output, output_size_bytes, cudaMemcpyDeviceToHost));
	
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_output));
	return(0);
}

