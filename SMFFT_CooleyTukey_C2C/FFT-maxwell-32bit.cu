#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "debug.h"
#include "timer.h"
#include "utils_cuda.h"
#include <stdio.h>

#define NREUSES 100
#define NCUDABLOCKS 1000

int device=0;

class FFT_Params {
public:
	static const int fft_exp = -1;
	static const int fft_length = -1;
	static const int warp = 32;
};

class FFT_256 : public FFT_Params {
	public:
	static const int fft_exp = 8;
	static const int fft_length = 256;
	static const int fft_length_quarter = 64;
	static const int fft_length_half = 128;
	static const int fft_length_three_quarters = 192;
};

class FFT_512 : public FFT_Params {
	public:
	static const int fft_exp = 9;
	static const int fft_length = 512;
	static const int fft_length_quarter = 128;
	static const int fft_length_half = 256;
	static const int fft_length_three_quarters = 384;
};

class FFT_1024 : public FFT_Params {
	public:
	static const int fft_exp = 10;
	static const int fft_length = 1024;
	static const int fft_length_quarter = 256;
	static const int fft_length_half = 512;
	static const int fft_length_three_quarters = 768;
};

class FFT_2048 : public FFT_Params {
	public:
	static const int fft_exp = 11;
	static const int fft_length = 2048;
	static const int fft_length_quarter = 512;
	static const int fft_length_half = 1024;
	static const int fft_length_three_quarters = 1536;
};

class FFT_4096 : public FFT_Params {
	public:
	static const int fft_exp = 12;
	static const int fft_length = 4096;
	static const int fft_length_quarter = 1024;
	static const int fft_length_half = 2048;
	static const int fft_length_three_quarters = 3072;
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
	sincosf ( -6.283185308f*fdividef( (float) m, (float) N), &ctemp.y, &ctemp.x);
	return(ctemp);
}

__device__ __inline__ float shfl_xor(float *value, int par){
	#if (CUDART_VERSION >= 9000)
		return(__shfl_xor_sync(0xffffffff, (*value), par));
	#else
		return(__shfl_xor((*value), par));
	#endif
}

template<class const_params>
__device__ void do_CT_DIT_FFT_4way(float2 *s_input){
	float2 A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value;
	float2 W;
	float2 Aftemp, Bftemp, Cftemp, Dftemp;

	int local_id, warp_id;
	int j, m_param;
	int parity, itemp;
	int A_read_index, B_read_index, C_read_index, D_read_index;
	int PoT, PoTp1, q;
	
	local_id = threadIdx.x & (const_params::warp - 1);
	warp_id = threadIdx.x/const_params::warp;

	#ifdef TESTING
	int A_load_id, B_load_id, i, A_n, B_n;
	A_load_id = threadIdx.x;
	B_load_id = threadIdx.x + const_params::fft_length_quarter;
	A_n=threadIdx.x;
	B_n=threadIdx.x + const_params::fft_length_quarter;
	for(i=1; i<const_params::fft_exp; i++) {
		A_n >>= 1;
		B_n >>= 1;
		A_load_id <<= 1;
		A_load_id |= A_n & 1;
		B_load_id <<= 1;
		B_load_id |= B_n & 1;
    }
    A_load_id &= const_params::fft_length-1;
	B_load_id &= const_params::fft_length-1;
	
	//-----> Scrambling input
	A_DFT_value=s_input[A_load_id];
	B_DFT_value=s_input[A_load_id + 1];
	C_DFT_value=s_input[B_load_id];
	D_DFT_value=s_input[B_load_id + 1];
	__syncthreads();
	s_input[threadIdx.x]         = A_DFT_value;
	s_input[threadIdx.x + const_params::fft_length_half]   = B_DFT_value;
	s_input[threadIdx.x + const_params::fft_length_quarter]   = C_DFT_value;
	s_input[threadIdx.x + const_params::fft_length_three_quarters] = D_DFT_value;
	__syncthreads();
	#endif
	
	
	//-----> FFT
	//-->
	PoT=1;
	PoTp1=2;	

	//--> First iteration
	itemp=local_id&1;
	parity=(1-itemp*2);
	A_DFT_value=s_input[local_id + (warp_id<<2)*const_params::warp];
	B_DFT_value=s_input[local_id + (warp_id<<2)*const_params::warp + const_params::warp];
	C_DFT_value=s_input[local_id + (warp_id<<2)*const_params::warp + 2*const_params::warp];
	D_DFT_value=s_input[local_id + (warp_id<<2)*const_params::warp + 3*const_params::warp];
	
	__syncthreads();
	
	A_DFT_value.x=parity*A_DFT_value.x + shfl_xor(&A_DFT_value.x, 1);
	A_DFT_value.y=parity*A_DFT_value.y + shfl_xor(&A_DFT_value.y, 1);
	B_DFT_value.x=parity*B_DFT_value.x + shfl_xor(&B_DFT_value.x, 1);
	B_DFT_value.y=parity*B_DFT_value.y + shfl_xor(&B_DFT_value.y, 1);
	C_DFT_value.x=parity*C_DFT_value.x + shfl_xor(&C_DFT_value.x, 1);
	C_DFT_value.y=parity*C_DFT_value.y + shfl_xor(&C_DFT_value.y, 1);
	D_DFT_value.x=parity*D_DFT_value.x + shfl_xor(&D_DFT_value.x, 1);
	D_DFT_value.y=parity*D_DFT_value.y + shfl_xor(&D_DFT_value.y, 1);
	
	//--> Second through Fifth iteration (no synchronization)
	PoT=2;
	PoTp1=4;
	for(q=1;q<5;q++){
		m_param = (local_id & (PoTp1 - 1));
		itemp = m_param>>q;
		parity=((itemp<<1)-1);
		W = Get_W_value(PoTp1, itemp*m_param);
		
		Aftemp.x = W.x*A_DFT_value.x - W.y*A_DFT_value.y;
		Aftemp.y = W.x*A_DFT_value.y + W.y*A_DFT_value.x;
		Bftemp.x = W.x*B_DFT_value.x - W.y*B_DFT_value.y;
		Bftemp.y = W.x*B_DFT_value.y + W.y*B_DFT_value.x;
		Cftemp.x = W.x*C_DFT_value.x - W.y*C_DFT_value.y;
		Cftemp.y = W.x*C_DFT_value.y + W.y*C_DFT_value.x;
		Dftemp.x = W.x*D_DFT_value.x - W.y*D_DFT_value.y;
		Dftemp.y = W.x*D_DFT_value.y + W.y*D_DFT_value.x;
		
		A_DFT_value.x = Aftemp.x + parity*shfl_xor(&Aftemp.x,PoT);
		A_DFT_value.y = Aftemp.y + parity*shfl_xor(&Aftemp.y,PoT);
		B_DFT_value.x = Bftemp.x + parity*shfl_xor(&Bftemp.x,PoT);
		B_DFT_value.y = Bftemp.y + parity*shfl_xor(&Bftemp.y,PoT);
		C_DFT_value.x = Cftemp.x + parity*shfl_xor(&Cftemp.x,PoT);
		C_DFT_value.y = Cftemp.y + parity*shfl_xor(&Cftemp.y,PoT);
		D_DFT_value.x = Dftemp.x + parity*shfl_xor(&Dftemp.x,PoT);
		D_DFT_value.y = Dftemp.y + parity*shfl_xor(&Dftemp.y,PoT);	
		
		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
	
	itemp = local_id + (warp_id<<2)*const_params::warp;
	s_input[itemp]                        = A_DFT_value;
	s_input[itemp + const_params::warp]   = B_DFT_value;
	s_input[itemp + 2*const_params::warp] = C_DFT_value;
	s_input[itemp + 3*const_params::warp] = D_DFT_value;
	
	for(q=5;q<(const_params::fft_exp-1);q++){
		__syncthreads();
		m_param = threadIdx.x & (PoT - 1);
		j=threadIdx.x>>q;
		
		W=Get_W_value(PoTp1,m_param);

		A_read_index=j*(PoTp1<<1) + m_param;
		B_read_index=j*(PoTp1<<1) + m_param + PoT;
		C_read_index=j*(PoTp1<<1) + m_param + PoTp1;
		D_read_index=j*(PoTp1<<1) + m_param + 3*PoT;
		
		Aftemp = s_input[A_read_index];
		Bftemp = s_input[B_read_index];
		A_DFT_value.x=Aftemp.x + W.x*Bftemp.x - W.y*Bftemp.y;
		A_DFT_value.y=Aftemp.y + W.x*Bftemp.y + W.y*Bftemp.x;		
		B_DFT_value.x=Aftemp.x - W.x*Bftemp.x + W.y*Bftemp.y;
		B_DFT_value.y=Aftemp.y - W.x*Bftemp.y - W.y*Bftemp.x;
		
		Cftemp = s_input[C_read_index];
		Dftemp = s_input[D_read_index];
		C_DFT_value.x=Cftemp.x + W.x*Dftemp.x - W.y*Dftemp.y;
		C_DFT_value.y=Cftemp.y + W.x*Dftemp.y + W.y*Dftemp.x;		
		D_DFT_value.x=Cftemp.x - W.x*Dftemp.x + W.y*Dftemp.y;
		D_DFT_value.y=Cftemp.y - W.x*Dftemp.y - W.y*Dftemp.x;
		
		s_input[A_read_index]=A_DFT_value;
		s_input[B_read_index]=B_DFT_value;
		s_input[C_read_index]=C_DFT_value;
		s_input[D_read_index]=D_DFT_value;
		
		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
	
	//last iteration
	__syncthreads();
	m_param = threadIdx.x;
	
	W=Get_W_value(PoTp1,m_param);
    
	A_read_index = m_param;
	B_read_index = m_param + PoT;
	C_read_index = m_param + (PoT>>1);
	D_read_index = m_param + 3*(PoT>>1);
	
	Aftemp = s_input[A_read_index];
	Bftemp = s_input[B_read_index];
	A_DFT_value.x=Aftemp.x + W.x*Bftemp.x - W.y*Bftemp.y;
	A_DFT_value.y=Aftemp.y + W.x*Bftemp.y + W.y*Bftemp.x;		
	B_DFT_value.x=Aftemp.x - W.x*Bftemp.x + W.y*Bftemp.y;
	B_DFT_value.y=Aftemp.y - W.x*Bftemp.y - W.y*Bftemp.x;
	
	Cftemp = s_input[C_read_index];
	Dftemp = s_input[D_read_index];
	C_DFT_value.x=Cftemp.x + W.y*Dftemp.x + W.x*Dftemp.y;
	C_DFT_value.y=Cftemp.y + W.y*Dftemp.y - W.x*Dftemp.x;		
	D_DFT_value.x=Cftemp.x - W.y*Dftemp.x - W.x*Dftemp.y;
	D_DFT_value.y=Cftemp.y - W.y*Dftemp.y + W.x*Dftemp.x;
	
	s_input[A_read_index]=A_DFT_value;
	s_input[B_read_index]=B_DFT_value;
	s_input[C_read_index]=C_DFT_value;
	s_input[D_read_index]=D_DFT_value;	
}

template<class const_params>
__global__ void FFT_GPU_external_4way(float2 *d_input, float2* d_output) {
	__shared__ float2 s_input[const_params::fft_length];
	s_input[threadIdx.x]                                           = d_input[threadIdx.x + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_length_quarter]        = d_input[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_quarter];
	s_input[threadIdx.x + const_params::fft_length_half]           = d_input[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_half];
	s_input[threadIdx.x + const_params::fft_length_three_quarters] = d_input[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_three_quarters];
	
	__syncthreads();
	do_CT_DIT_FFT_4way<const_params>(s_input);
	
	__syncthreads();
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length]                                           = s_input[threadIdx.x];
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_quarter]        = s_input[threadIdx.x + const_params::fft_length_quarter];
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_half]           = s_input[threadIdx.x + const_params::fft_length_half];
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_three_quarters] = s_input[threadIdx.x + const_params::fft_length_three_quarters];
}

template<class const_params>
__global__ void FFT_GPU_multiple_4way(float2 *d_input, float2* d_output) {
	__shared__ float2 s_input[const_params::fft_length];
	s_input[threadIdx.x]                                           = d_input[threadIdx.x + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_length_quarter]        = d_input[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_quarter];
	s_input[threadIdx.x + const_params::fft_length_half]           = d_input[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_half];
	s_input[threadIdx.x + const_params::fft_length_three_quarters] = d_input[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_three_quarters];
	
	__syncthreads();
	for(int f=0;f<NREUSES;f++){
		do_CT_DIT_FFT_4way<const_params>(s_input);
	}
	__syncthreads();
	
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length]                                           = s_input[threadIdx.x];
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_quarter]        = s_input[threadIdx.x + const_params::fft_length_quarter];
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_half]           = s_input[threadIdx.x + const_params::fft_length_half];
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_three_quarters] = s_input[threadIdx.x + const_params::fft_length_three_quarters];
}

void FFT_init(){
	//---------> Specific nVidia stuff
	cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
}

int FFT_external_benchmark_4way(float2 *d_input, float2 *d_output, int FFT_size, int nFFTs, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize(nFFTs, 1, 1);
	dim3 blockSize(FFT_size/4, 1, 1);
	
	//---------> FFT part
	timer.Start();
	switch(FFT_size) {
		case 256:
			FFT_GPU_external_4way<FFT_256><<<gridSize, blockSize>>>(d_input, d_output);
			break;
			
		case 512:
			FFT_GPU_external_4way<FFT_512><<<gridSize, blockSize>>>(d_input, d_output);
			break;
		
		case 1024:
			FFT_GPU_external_4way<FFT_1024><<<gridSize, blockSize>>>(d_input, d_output);
			break;

		case 2048:
			FFT_GPU_external_4way<FFT_2048><<<gridSize, blockSize>>>(d_input, d_output);
			break;
			
		case 4096:
			FFT_GPU_external_4way<FFT_4096><<<gridSize, blockSize>>>(d_input, d_output);
			break;
		
		default : 
			printf("Error wrong FFT length!\n");
			break;
	}
	timer.Stop();
	
	*FFT_time += timer.Elapsed();
	return(0);
}

int FFT_multiple_benchmark_4way(float2 *d_input, float2 *d_output, int FFT_size, int nFFTs, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	int nBlocks = (int) (nFFTs/NREUSES);
	if(nBlocks == 0) {
		*FFT_time=-1;
		return(1);
	}
	dim3 gridSize_multiple(nBlocks, 1, 1);
	dim3 blockSize(FFT_size/4, 1, 1);
	
	//---------> FFT part
	timer.Start();
	switch(FFT_size) {
		case 256:
			FFT_GPU_multiple_4way<FFT_256><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			break;
			
		case 512:
			FFT_GPU_multiple_4way<FFT_512><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			break;
		
		case 1024:
			FFT_GPU_multiple_4way<FFT_1024><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			break;

		case 2048:
			FFT_GPU_multiple_4way<FFT_2048><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			break;
			
		case 4096:
			FFT_GPU_multiple_4way<FFT_4096><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			break;
		
		default :
			printf("Error wrong FFT length!\n");
			break;
	}
	timer.Stop();
	
	*FFT_time += timer.Elapsed();
	return(0);
}

// ***********************************************************************************
// ***********************************************************************************
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
	if (DEBUG) printf("  Running cuFFT...: \t\t");
	cufftHandle plan;
	cufftResult error;
	error = cufftPlan1d(&plan, FFT_size, CUFFT_C2C, nFFTs);
	if (CUFFT_SUCCESS != error){
		printf("CUFFT error: %d", error);
	}
	
	timer.Start();
	cufftExecC2C(plan, (cufftComplex *)d_input, (cufftComplex *)d_output, CUFFT_FORWARD);
	timer.Stop();
	time_cuFFT += timer.Elapsed();
	
	cufftDestroy(plan);
	if (DEBUG) printf("done in %g ms.\n", time_cuFFT);
	*single_ex_time = time_cuFFT;
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

int GPU_smFFT_4elements(float2 *h_input, float2 *h_output, int FFT_size, int nFFTs, int nRuns, double *single_ex_time, double *multi_ex_time){
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
		printf("Error: Not enough memory! Input data is too big for the device.\n");
		return(1);
	}
	
	//----------> Memory allocation
	float2 *d_input;
	float2 *d_output;
	checkCudaErrors(cudaMalloc((void **) &d_input,  sizeof(float2)*input_size));
	checkCudaErrors(cudaMalloc((void **) &d_output, sizeof(float2)*output_size));
	
	//---------> Measurements
	double time_FFT_external = 0, time_FFT_multiple = 0;
	
	checkCudaErrors(cudaGetLastError());
	
	//--------------------------------------------------
	//-------------------------> 4way
	if(MULTIPLE){
		if (DEBUG) printf("  Running shared memory FFT (Cooley-Tukey) 100 times per GPU kernel (eliminates device memory)... ");
		FFT_init();
		double total_time_FFT_multiple = 0;
		for(int f=0; f<nRuns; f++){
			//---> Copy Host -> Device
			checkCudaErrors(cudaMemcpy(d_input, h_input, input_size*sizeof(float2), cudaMemcpyHostToDevice));
			FFT_multiple_benchmark_4way(d_input, d_output, FFT_size, nFFTs, &total_time_FFT_multiple);
		}
		time_FFT_multiple = total_time_FFT_multiple/nRuns;
		if (DEBUG) printf("done in %g ms.\n", time_FFT_multiple);
		*multi_ex_time = time_FFT_multiple;
	}
	
	checkCudaErrors(cudaGetLastError());
	
	if(EXTERNAL){
		if (DEBUG) printf("  Running shared memory FFT (Cooley-Tukey)... ");
		FFT_init();
		double total_time_FFT_external = 0;
		for(int f=0; f<nRuns; f++){
			//---> Copy Host -> Device
			checkCudaErrors(cudaMemcpy(d_input, h_input, input_size*sizeof(float2), cudaMemcpyHostToDevice));
			FFT_external_benchmark_4way(d_input, d_output, FFT_size, nFFTs, &total_time_FFT_external);
		}
		time_FFT_external = total_time_FFT_external/nRuns;
		if (DEBUG) printf("done in %g ms.\n", time_FFT_external);
		*single_ex_time = time_FFT_external;
	}
	
	checkCudaErrors(cudaGetLastError());
	//-----------------------------------<
	//--------------------------------------------------
	printf("  SH FFT normal = %0.3f ms; SM FFT multiple times = %0.3f ms\n", time_FFT_external, time_FFT_multiple);
	
	//---------> Copy Device -> Host
	checkCudaErrors(cudaMemcpy(h_output, d_output, output_size*sizeof(float2), cudaMemcpyDeviceToHost));

	//---------> error check -----
	checkCudaErrors(cudaGetLastError());
	
	//---------> Feeing allocated resources
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_output));
	
	return(0);
}

