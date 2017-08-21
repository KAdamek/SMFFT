#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "debug.h"
#include "timer.h"
#include "utils_cuda.h"
#include "utils_file.h"

#include "params.h"

//#define REORDER
#define WARP 32
#define NREUSES 100
#define NCUDABLOCKS 1000

int device=0;

__device__ __inline__ float2 Get_W_value(int N, int m){
	float2 ctemp;
	ctemp.x=cosf( -2.0f*3.141592654f*fdividef( (float) m, (float) N) );
	ctemp.y=sinf( -2.0f*3.141592654f*fdividef( (float) m, (float) N) );
	return(ctemp);
}


__device__ void do_FFT(float2 *s_input){
	float2 A_DFT_value, B_DFT_value;
	float2 W;
	float2 Aftemp, Bftemp;

	int local_id, warp_id;
	int j, m_param, parity;
	int A_read_index, B_read_index;
	int PoT, PoTm1, q;
	
	local_id = threadIdx.x & (WARP - 1);
	warp_id = threadIdx.x/WARP;
	
	
	//-----> FFT
	//-->
	PoTm1 = (FFT_LENGTH>>1);
	PoT   = FFT_LENGTH;

	for(q=(FFT_EXP-1);q>4;q--){
		__syncthreads();
		m_param = threadIdx.x & (PoTm1 - 1);
		j=threadIdx.x>>q;
		
		W=Get_W_value(PoT, m_param);

		A_read_index=j*PoT + m_param;
		B_read_index=j*PoT + m_param + PoTm1;
		
		Aftemp = s_input[A_read_index];
		Bftemp = s_input[B_read_index];
		
		A_DFT_value.x = Aftemp.x + Bftemp.x;
		A_DFT_value.y = Aftemp.y + Bftemp.y;
		
		B_DFT_value.x = W.x*(Aftemp.x - Bftemp.x) - W.y*(Aftemp.y - Bftemp.y);
		B_DFT_value.y = W.x*(Aftemp.y - Bftemp.y) + W.y*(Aftemp.x - Bftemp.x);
		
		s_input[A_read_index]=A_DFT_value;
		s_input[B_read_index]=B_DFT_value;
		
		PoT=PoT>>1;
		PoTm1=PoTm1>>1;
	}

	__syncthreads();
	A_DFT_value=s_input[local_id + warp_id*2*WARP];
	B_DFT_value=s_input[local_id + warp_id*2*WARP + WARP];
	
	for(q=4;q>=0;q--){
		m_param = (local_id & (PoT - 1));
		j = m_param>>q;
		parity=(1-j*2);
		W = Get_W_value(PoT, j*(m_param-PoTm1));
		
		Aftemp.x = parity*A_DFT_value.x + __shfl_xor(A_DFT_value.x, PoTm1);
		Aftemp.y = parity*A_DFT_value.y + __shfl_xor(A_DFT_value.y, PoTm1);
		Bftemp.x = parity*B_DFT_value.x + __shfl_xor(B_DFT_value.x, PoTm1);
		Bftemp.y = parity*B_DFT_value.y + __shfl_xor(B_DFT_value.y, PoTm1);
		
		A_DFT_value.x = W.x*Aftemp.x - W.y*Aftemp.y; 
		A_DFT_value.y = W.x*Aftemp.y + W.y*Aftemp.x;
		B_DFT_value.x = W.x*Bftemp.x - W.y*Bftemp.y; 
		B_DFT_value.y = W.x*Bftemp.y + W.y*Bftemp.x;
		
		PoT=PoT>>1;
		PoTm1=PoTm1>>1;
	}
	
	s_input[local_id + warp_id*2*WARP] = A_DFT_value;
	s_input[local_id + warp_id*2*WARP + WARP] = B_DFT_value;
	
	__syncthreads();
	
	#ifdef REORDER
	int load_id, i, n;
	load_id = threadIdx.x;
	n=threadIdx.x;
	for(i=1; i<FFT_EXP; i++) {
		n >>= 1;
		load_id <<= 1;
		load_id |= n & 1;
    }
    load_id &= FFT_LENGTH-1;
	
	//-----> Scrambling input
	__syncthreads();
	A_DFT_value=s_input[load_id];
	B_DFT_value=s_input[load_id + 1];
	__syncthreads();
	s_input[threadIdx.x]     = A_DFT_value;
	s_input[threadIdx.x+FFT_LENGTH/2] = B_DFT_value;
	__syncthreads();
	#endif
}


__device__ void do_FFT_4way(float2 *s_input){
	float2 A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value;
	float2 W;
	float2 Aftemp, Bftemp, Cftemp, Dftemp;

	int local_id, warp_id;
	int j, m_param, parity;
	int A_read_index, B_read_index, C_read_index, D_read_index;
	int PoT, PoTm1, q;
	
	local_id = threadIdx.x & (WARP - 1);
	warp_id = threadIdx.x/WARP;
	
	
	//-----> FFT
	//-->
	PoTm1 = (FFT_LENGTH>>1);
	PoT   = FFT_LENGTH;
	
	//Highest iteration
	m_param = threadIdx.x;
	j=0;
	A_read_index = m_param;
	B_read_index = m_param + PoTm1;
	C_read_index = m_param + (PoTm1>>1);
	D_read_index = m_param + 3*(PoTm1>>1);
	
	W=Get_W_value(PoT, m_param);
	
	Aftemp = s_input[A_read_index];
	Bftemp = s_input[B_read_index];
	Cftemp = s_input[C_read_index];
	Dftemp = s_input[D_read_index];
	
	A_DFT_value.x = Aftemp.x + Bftemp.x;
	A_DFT_value.y = Aftemp.y + Bftemp.y;
	B_DFT_value.x = W.x*(Aftemp.x - Bftemp.x) - W.y*(Aftemp.y - Bftemp.y);
	B_DFT_value.y = W.x*(Aftemp.y - Bftemp.y) + W.y*(Aftemp.x - Bftemp.x);
	
	C_DFT_value.x = Cftemp.x + Dftemp.x;
	C_DFT_value.y = Cftemp.y + Dftemp.y;
	D_DFT_value.x = W.y*(Cftemp.x - Dftemp.x) + W.x*(Cftemp.y - Dftemp.y);
	D_DFT_value.y = W.y*(Cftemp.y - Dftemp.y) - W.x*(Cftemp.x - Dftemp.x);
	
	s_input[A_read_index]=A_DFT_value;
	s_input[B_read_index]=B_DFT_value;
	s_input[C_read_index]=C_DFT_value;
	s_input[D_read_index]=D_DFT_value;
	
	PoT=PoT>>1;
	PoTm1=PoTm1>>1;
	
	for(q=(FFT_EXP-2);q>4;q--){
		__syncthreads();
		m_param = threadIdx.x & (PoTm1 - 1);
		j=threadIdx.x>>q;
		
		W=Get_W_value(PoT, m_param);

		A_read_index=j*(PoT<<1) + m_param;
		B_read_index=j*(PoT<<1) + m_param + PoTm1;
		C_read_index=j*(PoT<<1) + m_param + PoT;
		D_read_index=j*(PoT<<1) + m_param + 3*PoTm1;
		
		Aftemp = s_input[A_read_index];
		Bftemp = s_input[B_read_index];
		Cftemp = s_input[C_read_index];
		Dftemp = s_input[D_read_index];
		
		A_DFT_value.x = Aftemp.x + Bftemp.x;
		A_DFT_value.y = Aftemp.y + Bftemp.y;
		C_DFT_value.x = Cftemp.x + Dftemp.x;
		C_DFT_value.y = Cftemp.y + Dftemp.y;
		
		B_DFT_value.x = W.x*(Aftemp.x - Bftemp.x) - W.y*(Aftemp.y - Bftemp.y);
		B_DFT_value.y = W.x*(Aftemp.y - Bftemp.y) + W.y*(Aftemp.x - Bftemp.x);
		D_DFT_value.x = W.x*(Cftemp.x - Dftemp.x) - W.y*(Cftemp.y - Dftemp.y);
		D_DFT_value.y = W.x*(Cftemp.y - Dftemp.y) + W.y*(Cftemp.x - Dftemp.x);
		
		s_input[A_read_index]=A_DFT_value;
		s_input[B_read_index]=B_DFT_value;
		s_input[C_read_index]=C_DFT_value;
		s_input[D_read_index]=D_DFT_value;
		
		PoT=PoT>>1;
		PoTm1=PoTm1>>1;
	}

	__syncthreads();
	j = local_id + (warp_id<<2)*WARP;
	A_DFT_value = s_input[j];
	B_DFT_value = s_input[j + WARP];
	C_DFT_value = s_input[j + 2*WARP];
	D_DFT_value = s_input[j + 3*WARP];
	
	for(q=4;q>=0;q--){
		m_param = (local_id & (PoT - 1));
		j = m_param>>q;
		parity=(1-j*2);
		W = Get_W_value(PoT, j*(m_param-PoTm1));
		
		Aftemp.x = parity*A_DFT_value.x + __shfl_xor(A_DFT_value.x, PoTm1);
		Aftemp.y = parity*A_DFT_value.y + __shfl_xor(A_DFT_value.y, PoTm1);
		Bftemp.x = parity*B_DFT_value.x + __shfl_xor(B_DFT_value.x, PoTm1);
		Bftemp.y = parity*B_DFT_value.y + __shfl_xor(B_DFT_value.y, PoTm1);
		Cftemp.x = parity*C_DFT_value.x + __shfl_xor(C_DFT_value.x, PoTm1);
		Cftemp.y = parity*C_DFT_value.y + __shfl_xor(C_DFT_value.y, PoTm1);
		Dftemp.x = parity*D_DFT_value.x + __shfl_xor(D_DFT_value.x, PoTm1);
		Dftemp.y = parity*D_DFT_value.y + __shfl_xor(D_DFT_value.y, PoTm1);
		
		A_DFT_value.x = W.x*Aftemp.x - W.y*Aftemp.y; 
		A_DFT_value.y = W.x*Aftemp.y + W.y*Aftemp.x;
		B_DFT_value.x = W.x*Bftemp.x - W.y*Bftemp.y; 
		B_DFT_value.y = W.x*Bftemp.y + W.y*Bftemp.x;
		C_DFT_value.x = W.x*Cftemp.x - W.y*Cftemp.y; 
		C_DFT_value.y = W.x*Cftemp.y + W.y*Cftemp.x;
		D_DFT_value.x = W.x*Dftemp.x - W.y*Dftemp.y; 
		D_DFT_value.y = W.x*Dftemp.y + W.y*Dftemp.x;
		
		PoT=PoT>>1;
		PoTm1=PoTm1>>1;
	}
	
	j = local_id + (warp_id<<2)*WARP;
	s_input[j]          = A_DFT_value;
	s_input[j + WARP]   = B_DFT_value;
	s_input[j + 2*WARP] = C_DFT_value;
	s_input[j + 3*WARP] = D_DFT_value;
	
	__syncthreads();
	
	#ifdef REORDER
	__syncthreads();
	int A_load_id, B_load_id, i, A_n, B_n;
	A_load_id = threadIdx.x;
	B_load_id = threadIdx.x + FFT_LENGTH/4;
	A_n=threadIdx.x;
	B_n=threadIdx.x + FFT_LENGTH/4;
	for(i=1; i<FFT_EXP; i++) {
		A_n >>= 1;
		B_n >>= 1;
		A_load_id <<= 1;
		A_load_id |= A_n & 1;
		B_load_id <<= 1;
		B_load_id |= B_n & 1;
    }
    A_load_id &= FFT_LENGTH-1;
	B_load_id &= FFT_LENGTH-1;
	
	//-----> Scrambling input
	A_DFT_value=s_input[A_load_id];
	B_DFT_value=s_input[A_load_id + 1];
	C_DFT_value=s_input[B_load_id];
	D_DFT_value=s_input[B_load_id + 1];
	__syncthreads();
	s_input[threadIdx.x]         = A_DFT_value;
	s_input[threadIdx.x + FFT_LENGTH/2]   = B_DFT_value;
	s_input[threadIdx.x + FFT_LENGTH/4]   = C_DFT_value;
	s_input[threadIdx.x + 3*FFT_LENGTH/4] = D_DFT_value;
	__syncthreads();
	#endif
}




__global__ void FFT_GPU_external(float2 *d_input, float2* d_output) {
	extern __shared__ float2 s_input[];
	s_input[threadIdx.x]=d_input[threadIdx.x + blockIdx.x*FFT_LENGTH];
	s_input[threadIdx.x + FFT_LENGTH/2]=d_input[threadIdx.x + blockIdx.x*FFT_LENGTH + FFT_LENGTH/2];
	
	__syncthreads();
	do_FFT(s_input);
	
	__syncthreads();
	d_output[threadIdx.x + blockIdx.x*FFT_LENGTH]=s_input[threadIdx.x];
	d_output[threadIdx.x + blockIdx.x*FFT_LENGTH + FFT_LENGTH/2]=s_input[threadIdx.x + FFT_LENGTH/2];
}

__global__ void FFT_GPU_external_4way(float2 *d_input, float2* d_output) {
	extern __shared__ float2 s_input[];
	s_input[threadIdx.x]=d_input[threadIdx.x + blockIdx.x*FFT_LENGTH];
	s_input[threadIdx.x + (FFT_LENGTH>>2)]=d_input[threadIdx.x + blockIdx.x*FFT_LENGTH + (FFT_LENGTH>>2)];
	s_input[threadIdx.x + (FFT_LENGTH>>1)]=d_input[threadIdx.x + blockIdx.x*FFT_LENGTH + (FFT_LENGTH>>1)];
	s_input[threadIdx.x + 3*(FFT_LENGTH>>2)]=d_input[threadIdx.x + blockIdx.x*FFT_LENGTH + 3*(FFT_LENGTH>>2)];
	
	__syncthreads();
	do_FFT_4way(s_input);
	
	__syncthreads();
	d_output[threadIdx.x + blockIdx.x*FFT_LENGTH]=s_input[threadIdx.x];
	d_output[threadIdx.x + blockIdx.x*FFT_LENGTH + (FFT_LENGTH>>2)]=s_input[threadIdx.x + (FFT_LENGTH>>2)];
	d_output[threadIdx.x + blockIdx.x*FFT_LENGTH + (FFT_LENGTH>>1)]=s_input[threadIdx.x + (FFT_LENGTH>>1)];
	d_output[threadIdx.x + blockIdx.x*FFT_LENGTH + 3*(FFT_LENGTH>>2)]=s_input[threadIdx.x + 3*(FFT_LENGTH>>2)];
}

__global__ void FFT_GPU_multiple(float2 *d_input, float2* d_output) {
	extern __shared__ float2 s_input[];
	s_input[threadIdx.x]=d_input[threadIdx.x + blockIdx.x*FFT_LENGTH];
	s_input[threadIdx.x + FFT_LENGTH/2]=d_input[threadIdx.x + blockIdx.x*FFT_LENGTH + FFT_LENGTH/2];
	
	__syncthreads();
	for(int f=0;f<NREUSES;f++){
		do_FFT(s_input);
	}
	
	__syncthreads();
	d_output[threadIdx.x + blockIdx.x*FFT_LENGTH]=s_input[threadIdx.x];
	d_output[threadIdx.x + blockIdx.x*FFT_LENGTH + FFT_LENGTH/2]=s_input[threadIdx.x + FFT_LENGTH/2];
}

__global__ void FFT_GPU_multiple_4way(float2 *d_input, float2* d_output) {
	extern __shared__ float2 s_input[];
	s_input[threadIdx.x]=d_input[threadIdx.x + blockIdx.x*FFT_LENGTH];
	s_input[threadIdx.x + (FFT_LENGTH>>2)]=d_input[threadIdx.x + blockIdx.x*FFT_LENGTH + (FFT_LENGTH>>2)];
	s_input[threadIdx.x + (FFT_LENGTH>>1)]=d_input[threadIdx.x + blockIdx.x*FFT_LENGTH + (FFT_LENGTH>>1)];
	s_input[threadIdx.x + 3*(FFT_LENGTH>>2)]=d_input[threadIdx.x + blockIdx.x*FFT_LENGTH + 3*(FFT_LENGTH>>2)];
	
	__syncthreads();
	for(int f=0;f<NREUSES;f++){
		do_FFT_4way(s_input);
	}
	__syncthreads();
	
	d_output[threadIdx.x + blockIdx.x*FFT_LENGTH]=s_input[threadIdx.x];
	d_output[threadIdx.x + blockIdx.x*FFT_LENGTH + (FFT_LENGTH>>2)]=s_input[threadIdx.x + (FFT_LENGTH>>2)];
	d_output[threadIdx.x + blockIdx.x*FFT_LENGTH + (FFT_LENGTH>>1)]=s_input[threadIdx.x + (FFT_LENGTH>>1)];
	d_output[threadIdx.x + blockIdx.x*FFT_LENGTH + 3*(FFT_LENGTH>>2)]=s_input[threadIdx.x + 3*(FFT_LENGTH>>2)];
}


int Max_columns_in_memory_shared(int nSamples, int nSpectra) {
	long int nColumns,maxgrid_x;

	size_t free_mem,total_mem;
	cudaDeviceProp devProp;
	
	checkCudaErrors(cudaSetDevice(device));
	checkCudaErrors(cudaGetDeviceProperties(&devProp,device));
	maxgrid_x = devProp.maxGridSize[0];
	cudaMemGetInfo(&free_mem,&total_mem);
	
	nColumns=((long int) free_mem)/(2.0*sizeof(float2)*nSamples);
	if(nColumns>maxgrid_x) nColumns=maxgrid_x;
	nColumns=(int) nColumns*0.9;
	return(nColumns);
}


void FFT_init(){
	//---------> Specific nVidia stuff
	cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
}

void FFT_external_benchmark(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	int nCUDAblocks_x=nSpectra;
	int nCUDAblocks_y=1; //Head size
	
	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);	//nCUDAblocks_y goes through spectra
	dim3 blockSize(nSamples/2, 1, 1); 				//nCUDAblocks_x goes through channels
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_external<<<gridSize, blockSize,nSamples*8>>>( d_input, d_output);
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_external_benchmark_4way(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	int nCUDAblocks_x=nSpectra;
	int nCUDAblocks_y=1; //Head size
	
	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);	//nCUDAblocks_y goes through spectra
	dim3 blockSize(nSamples/4, 1, 1); 				//nCUDAblocks_x goes through channels
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_external_4way<<<gridSize, blockSize,nSamples*8>>>( d_input, d_output);
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_multiple_benchmark(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize_multiple(NCUDABLOCKS, 1, 1);	//nCUDAblocks_y goes through spectra
	dim3 blockSize(nSamples/2, 1, 1); 				//nCUDAblocks_x goes through channels
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_multiple<<<gridSize_multiple, blockSize,nSamples*8>>>( d_input, d_output);
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_multiple_benchmark_4way(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize_multiple(NCUDABLOCKS, 1, 1);	//nCUDAblocks_y goes through spectra
	dim3 blockSize(nSamples/4, 1, 1); 				//nCUDAblocks_x goes through channels
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_multiple_4way<<<gridSize_multiple, blockSize,nSamples*8>>>( d_input, d_output);
	timer.Stop();
	*FFT_time += timer.Elapsed();
}


// ***********************************************************************************
// ***********************************************************************************
// ***********************************************************************************


int GPU_FFT(float2 *input, float2 *output, int nSamples, int nSpectra, int nRuns){
	//---------> Initial nVidia stuff
	int devCount;
	cudaDeviceProp devProp;
	size_t free_mem,total_mem;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if (DEBUG) {
		printf("\nThere are %d devices.", devCount);
		for (int i = 0; i < devCount; i++){
			checkCudaErrors(cudaGetDeviceProperties(&devProp,i));
			printf("\n\t Using device:\t\t\t%s\n", devProp.name);
			printf("\n\t Max grid size:\t\t\t%d\n", devProp.maxGridSize[1]);
			printf("\n\t Shared mem per block:\t\t%d\n", devProp.sharedMemPerBlock);
		}
	}
	checkCudaErrors(cudaSetDevice(device));
	checkCudaErrors(cudaGetDeviceProperties(&devProp,device));
	
	cudaMemGetInfo(&free_mem,&total_mem);
	if(DEBUG) printf("\nDevice has %ld MB of total memory, which %ld MB is available.\n", (long int) total_mem/(1000*1000), (long int) free_mem/(1000*1000));
	
	//---------> Measurements
	double transfer_in, transfer_out, FFT_time, FFT_external_time, FFT_multiple_time, FFT_multiple_reuse_time,cuFFT_time,FFT_multiple_reuse_registers_time;
	double FFT_external_time_total, FFT_multiple_time_total;
	GpuTimer timer; // if set before set device getting errors - invalid handle  
	
	
	//------------------------------------------------------------------------------
	//---------> Shared memory kernel
	transfer_in=0.0; transfer_out=0.0; FFT_time=0.0; FFT_external_time=0.0; FFT_multiple_time=0.0; FFT_multiple_reuse_time=0.0; cuFFT_time=0.0; FFT_multiple_reuse_registers_time=0.0;
	FFT_external_time_total=0.0; FFT_multiple_time_total=0.0;
	
	//---------> Spectra
	int maxColumns,Sremainder,nRepeats,Spectra_to_allocate;
	maxColumns=Max_columns_in_memory_shared(nSamples,nSpectra); // Maximum number of columns which fits into memory
	nRepeats=(int) (nSpectra/maxColumns);
	Sremainder=nSpectra-nRepeats*maxColumns;
	Spectra_to_allocate=Sremainder;
	if(nRepeats>0) Spectra_to_allocate=maxColumns;
	if(nRepeats>0) {printf("Array is too big. Choose smaller number of FFTs\n"); exit(1);}
	
	if(Spectra_to_allocate>maxColumns) {printf("Remainder is greater then maxColumns");exit(2);}
	if (DEBUG) printf("Maximum number of spectra %d which is %e MB \n",maxColumns, (double) (maxColumns*nSamples*sizeof(float)/(1000.0*1000.0))   );
	if (DEBUG) printf("nColumns is split into %d chunks of %d spectra and into remainder of %d spectra.\n",nRepeats,maxColumns,Sremainder);
	if (DEBUG) printf("Number of columns execute is %d.\n",Sremainder);
	
	//---------> Channels
	//if( nSamples%32!=0) {printf("Number of channels must be divisible by 32"); exit(2);}
	
	//---------> Memory allocation
	if (DEBUG) printf("Device memory allocation...: \t\t");
	int input_size=nSamples*Spectra_to_allocate;
	int output_size=nSamples*Spectra_to_allocate;
	float2 *d_output;
	float2 *d_input;
	timer.Start();
	checkCudaErrors(cudaMalloc((void **) &d_input,  sizeof(float2)*input_size));
	checkCudaErrors(cudaMalloc((void **) &d_output, sizeof(float2)*output_size));
	timer.Stop();
	if (DEBUG) printf("done in %g ms.\n", timer.Elapsed());

	//---------> FFT calculation
	for (int r = 0; r < nRepeats; r++){
	}
	if (Sremainder>0){
		//-----> Copy chunk of input data to a device
		if (DEBUG) printf("Transferring data into device memory...: \t\t");
		timer.Start();
		checkCudaErrors(cudaMemcpy(d_input, &input[nRepeats*output_size], (Sremainder)*nSamples*sizeof(float2), cudaMemcpyHostToDevice));
		timer.Stop();
		transfer_in+=timer.Elapsed();
		if (DEBUG) printf("done in %g ms.\n", timer.Elapsed());
	
		//-----> Compute FFT on the chunk
		if(CUFFT){
			//---------> FFT
			cufftHandle plan;
			cufftResult error;
			error = cufftPlan1d(&plan, nSamples, CUFFT_C2C, Sremainder);
			if (CUFFT_SUCCESS != error){
				printf("CUFFT error: %d", error);
			}
			
			timer.Start();
			cufftExecC2C(plan, (cufftComplex *)d_input, (cufftComplex *)d_output, CUFFT_FORWARD);
			timer.Stop();
			cuFFT_time += timer.Elapsed();
			
			cufftDestroy(plan);
		}
		
		
		//------------------------------> 2way (normal)
		if(MULTIPLE){
			if (DEBUG) printf("\nApplying MULTIPLE FFT...: \t\t");
			FFT_init();
			FFT_multiple_time_total = 0;
			for(int f=0; f<nRuns; f++){
				checkCudaErrors(cudaMemcpy(d_input, &input[nRepeats*output_size], (Sremainder)*nSamples*sizeof(float2), cudaMemcpyHostToDevice));
				FFT_multiple_benchmark(d_input, d_output, nSamples, Sremainder, &FFT_multiple_time_total);
			}
			FFT_multiple_time = FFT_multiple_time_total/nRuns;
			if (DEBUG) printf("done in %g ms.\n", FFT_multiple_time);
		}
		
		
		if(EXTERNAL){
			if (DEBUG) printf("\nApplying EXTERNAL FFT...: \t\t");
			FFT_init();
			FFT_external_time_total = 0;
			for(int f=0; f<nRuns; f++){
				checkCudaErrors(cudaMemcpy(d_input, &input[nRepeats*output_size], (Sremainder)*nSamples*sizeof(float2), cudaMemcpyHostToDevice));
				FFT_external_benchmark(d_input, d_output, nSamples, Sremainder, &FFT_external_time_total);
			}
			FFT_external_time = FFT_external_time_total/nRuns;
			if (DEBUG) printf("done in %g ms.\n", FFT_external_time);
		}
		//----------------------------------<
		
		
		//-------------------------> 4way
		if(MULTIPLE){
			if (DEBUG) printf("\nApplying MULTIPLE FFT 4way...: \t\t");
			FFT_init();
			FFT_multiple_time_total = 0;
			for(int f=0; f<nRuns; f++){
				checkCudaErrors(cudaMemcpy(d_input, &input[nRepeats*output_size], (Sremainder)*nSamples*sizeof(float2), cudaMemcpyHostToDevice));
				FFT_multiple_benchmark_4way(d_input, d_output, nSamples, Sremainder, &FFT_multiple_time_total);
			}
			FFT_multiple_time = FFT_multiple_time_total/nRuns;
			if (DEBUG) printf("done in %g ms.\n", FFT_multiple_time);
		}
		
		if(EXTERNAL){
			if (DEBUG) printf("\nApplying EXTERNAL FFT 4way...: \t\t");
			FFT_init();
			FFT_external_time_total = 0;
			for(int f=0; f<nRuns; f++){
				checkCudaErrors(cudaMemcpy(d_input, &input[nRepeats*output_size], (Sremainder)*nSamples*sizeof(float2), cudaMemcpyHostToDevice));
				FFT_external_benchmark_4way(d_input, d_output, nSamples, Sremainder, &FFT_external_time_total);
			}
			FFT_external_time = FFT_external_time_total/nRuns;
			if (DEBUG) printf("done in %g ms.\n", FFT_external_time);
		}
		//-----------------------------------<
		
		
		//-----> Copy chunk of output data to host
		if (DEBUG) printf("Transferring data to host...: \t\t");
		timer.Start();
		checkCudaErrors(cudaMemcpy( &output[nRepeats*output_size], d_output, (Sremainder)*nSamples*sizeof(float2), cudaMemcpyDeviceToHost));
		timer.Stop();
		transfer_out+=timer.Elapsed();
		if (DEBUG) printf("done in %g ms.\n", timer.Elapsed());
	}

	

	//---------> error check -----
	checkCudaErrors(cudaGetLastError());
	
	//---------> Feeing allocated resources
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_output));
	
	if (DEBUG || WRITE) printf("nSpectra:%d; nSamples:%d cuFFT:%0.3f ms; FFT external:%0.3f ms; FFT multiple:%0.3f ms; \n",nSpectra,nSamples,cuFFT_time, FFT_external_time, FFT_multiple_time);	
	
	if (WRITE){ 
		char str[200];
		sprintf(str,"GPU-polyphase-precisioncontrol.dat");
		if (DEBUG) printf("\n Write results into file...\t");
		save_time(str, nSpectra,nSamples, cuFFT_time, FFT_time, FFT_external_time, FFT_multiple_time, FFT_multiple_reuse_time, FFT_multiple_reuse_registers_time, transfer_in, transfer_out);
		if (DEBUG) printf("\t done.\n-------------------------------------\n");
	}
	
}
