#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "debug.h"
#include "timer.h"
#include "utils_cuda.h"
#include "utils_file.h"

#include "params.h"


#define WARP 32
#define NREUSES 100
#define NCUDABLOCKS 1000
#define BANKCONFLICT 7 // for N=1024

int device=0;

__device__ __inline__ float2 Get_W_value(int N, int m){
	float2 ctemp;
	ctemp.x=cosf( -2.0f*3.141592654f*fdividef( (float) m, (float) N) );
	ctemp.y=sinf( -2.0f*3.141592654f*fdividef( (float) m, (float) N) );
	return(ctemp);
}


__device__ void do_FFT(float2 *s_input, int N, int bits){
	float2 A_DFT_value, B_DFT_value;
	float2 W;
	float2 ftemp, ftemp2;

	int local_id, warp_id;
	int j, m_param;
	int load_id, i, n;
	int parity, itemp;
	int A_read_index,B_read_index;
	int PoT, PoTp1, q;
	
	local_id = threadIdx.x & (WARP - 1);
	warp_id = threadIdx.x/WARP;
	
	load_id = threadIdx.x;
	n=threadIdx.x;
  	for(i=1; i<bits; i++) {
		n >>= 1;
		load_id <<= 1;
		load_id |= n & 1;
  	}
  	load_id &= N-1;
	
	//-----> Scrambling input
	A_read_index=load_id>>BANKCONFLICT;
	A_DFT_value=s_input[load_id + A_read_index];
	
	B_read_index=(load_id + 1)>>BANKCONFLICT;
	B_DFT_value=s_input[load_id + 1 + B_read_index];
	__syncthreads();
	s_input[threadIdx.x]   = A_DFT_value;
	s_input[threadIdx.x+N/2] = B_DFT_value;
	__syncthreads();
	
	//-----> FFT
	//-->
	PoT=1;
	PoTp1=2;	

	//--> First iteration
	itemp=local_id&1;
	parity=(1-itemp*2);
	A_DFT_value=s_input[local_id + warp_id*2*WARP];
	B_DFT_value=s_input[local_id + warp_id*2*WARP + WARP];
	
	A_DFT_value.x=parity*A_DFT_value.x + __shfl(A_DFT_value.x,local_id + parity);
	A_DFT_value.y=parity*A_DFT_value.y + __shfl(A_DFT_value.y,local_id + parity);
	
	B_DFT_value.x=parity*B_DFT_value.x + __shfl(B_DFT_value.x,local_id + parity);
	B_DFT_value.y=parity*B_DFT_value.y + __shfl(B_DFT_value.y,local_id + parity);
	
	
	//--> Second through Fifth iteration (no synchronization)
	PoT=2;
	PoTp1=4;
	for(q=1;q<5;q++){
		m_param = (local_id & (PoTp1 - 1));
		itemp=m_param>>q;

		W=Get_W_value(PoTp1,m_param);
		
		ftemp2.x=__shfl_xor(A_DFT_value.x,(1-itemp)*PoT);
		ftemp2.y=__shfl_xor(A_DFT_value.y,(1-itemp)*PoT);					
		A_DFT_value.x=__shfl_xor(A_DFT_value.x,itemp*PoT) + W.x*ftemp2.x-W.y*ftemp2.y;
		A_DFT_value.y=__shfl_xor(A_DFT_value.y,itemp*PoT) + W.x*ftemp2.y+W.y*ftemp2.x;
		
		ftemp.x=__shfl_xor(B_DFT_value.x,(1-itemp)*PoT);
		ftemp.y=__shfl_xor(B_DFT_value.y,(1-itemp)*PoT);					
		B_DFT_value.x=__shfl_xor(B_DFT_value.x,itemp*PoT) + W.x*ftemp.x - W.y*ftemp.y;
		B_DFT_value.y=__shfl_xor(B_DFT_value.y,itemp*PoT) + W.x*ftemp.y + W.y*ftemp.x;	
		
		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
	
	__syncthreads();
	s_input[local_id + warp_id*2*WARP]=A_DFT_value;
	s_input[local_id + warp_id*2*WARP + WARP]=B_DFT_value;
	__syncthreads();
	
	//--> Remaining iterations
	for(q=5;q<bits;q++){
		m_param = threadIdx.x & (PoT - 1);
		j=threadIdx.x>>q;
		
		W=Get_W_value(PoTp1,m_param);

		A_read_index=j*PoTp1 + m_param;
		B_read_index=j*PoTp1 + m_param + PoT;
		
		ftemp  = s_input[A_read_index];
		ftemp2 = s_input[B_read_index];
		
		A_DFT_value.x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		A_DFT_value.y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		B_DFT_value.x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		B_DFT_value.y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;
		
		__syncthreads();
		s_input[A_read_index]=A_DFT_value;
		s_input[B_read_index]=B_DFT_value;
		__syncthreads();
		
		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
}


__global__ void FFT_GPU_external(float2 *d_input, float2* d_output, int N, int bits) {
	extern __shared__ float2 s_input[];
	int pos;
	
	pos=threadIdx.x>>BANKCONFLICT;
	s_input[threadIdx.x + pos]=d_input[threadIdx.x + blockIdx.x*N];
	pos=(threadIdx.x + N/2)>>BANKCONFLICT;
	s_input[threadIdx.x + N/2 + pos]=d_input[threadIdx.x + blockIdx.x*N + N/2];
	
	__syncthreads();
	do_FFT(s_input,N,bits);
	
	__syncthreads();
	d_output[threadIdx.x + blockIdx.x*N]=s_input[threadIdx.x];
	d_output[threadIdx.x + blockIdx.x*N + N/2]=s_input[threadIdx.x + N/2];
}


__global__ void FFT_GPU_multiple(float2 *d_input, float2* d_output, int N, int bits) {
	extern __shared__ float2 s_input[];
	s_input[threadIdx.x]=d_input[threadIdx.x + blockIdx.x*N];
	s_input[threadIdx.x + N/2]=d_input[threadIdx.x + blockIdx.x*N + N/2];
	
	__syncthreads();
	for(int f=0;f<NREUSES;f++){
		do_FFT(s_input,N,bits);
	}
	
	__syncthreads();
	d_output[threadIdx.x + blockIdx.x*N]=s_input[threadIdx.x];
	d_output[threadIdx.x + blockIdx.x*N + N/2]=s_input[threadIdx.x + N/2];
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
	int itemp;
	//---------> CUDA block and CUDA grid parameters
	int nCUDAblocks_x=nSpectra;
	int nCUDAblocks_y=1; //Head size
	
	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);	//nCUDAblocks_y goes through spectra
	dim3 blockSize(nSamples/2, 1, 1); 				//nCUDAblocks_x goes through channels
	itemp=nSamples>>BANKCONFLICT;
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_external<<<gridSize, blockSize,(nSamples+itemp)*8>>>( d_input, d_output, nSamples,round(log(nSamples)/log(2)));
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_multiple_benchmark(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	int itemp;
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize_multiple(NCUDABLOCKS, 1, 1);	//nCUDAblocks_y goes through spectra
	dim3 blockSize(nSamples/2, 1, 1); 				//nCUDAblocks_x goes through channels
	itemp=nSamples>>BANKCONFLICT;
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_multiple<<<gridSize_multiple, blockSize,(nSamples+itemp)*8>>>( d_input, d_output, nSamples,round(log(nSamples)/log(2)));
	timer.Stop();
	*FFT_time += timer.Elapsed();
}



// ***********************************************************************************
// ***********************************************************************************
// ***********************************************************************************


int GPU_FFT(float2 *h_input, float2 *h_output, int nSamples, int nSpectra, int inverse){
	//---------> Initial nVidia stuff
	int devCount;
	size_t free_mem,total_mem;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	checkCudaErrors(cudaSetDevice(device));
	
	cudaMemGetInfo(&free_mem,&total_mem);
	if(DEBUG) printf("\nDevice has %ld MB of total memory, which %ld MB is available.\n", (long int) total_mem/(1000*1000), (long int) free_mem/(1000*1000));
	
	//---------> Checking memory
	int nElements=nSamples*nSpectra;
	int input_size=nElements;
	int output_size=nElements;
	
	float free_memory = (float) free_mem/(1024.0*1024.0);
	float memory_required=((2*input_size + 2*output_size)*sizeof(float))/(1024.0*1024.0);
	if(DEBUG) printf("DEBUG: Device has %0.3f MB of total memory, which %0.3f MB is available. Memory required %0.3f MB\n", (float) total_mem/(1024.0*1024.0), free_memory ,memory_required);
	if(memory_required>free_memory) {printf("\n \n Array is too big for the device! \n \n"); return(-3);}
		
	//---------> Measurements
	double transfer_in, transfer_out, FFT_time, FFT_external_time, FFT_multiple_time, FFT_multiple_reuse_time,cuFFT_time,FFT_multiple_reuse_registers_time;
	GpuTimer timer; // if set before set device getting errors - invalid handle  
	
	
	//------------------------------------------------------------------------------
	//---------> Shared memory kernel
	transfer_in=0.0; transfer_out=0.0; FFT_time=0.0; FFT_external_time=0.0; FFT_multiple_time=0.0; FFT_multiple_reuse_time=0.0; cuFFT_time=0.0; FFT_multiple_reuse_registers_time=0.0;
	
	//---------> Memory allocation
	if (DEBUG) printf("Device memory allocation...: \t\t");
	float2 *d_output;
	float2 *d_input;
	timer.Start();
	checkCudaErrors(cudaMalloc((void **) &d_input,  sizeof(float2)*input_size));
	checkCudaErrors(cudaMalloc((void **) &d_output, sizeof(float2)*output_size));
	timer.Stop();
	if (DEBUG) printf("done in %g ms.\n", timer.Elapsed());

	//---------> FFT calculation
	if (DEBUG) printf("Transferring data to device...: \t");
	timer.Start();
	checkCudaErrors(cudaMemcpy(d_input, h_input, input_size*sizeof(float2), cudaMemcpyHostToDevice));
	timer.Stop();
	transfer_in+=timer.Elapsed();
	if (DEBUG) printf("done in %g ms.\n", timer.Elapsed());
	
	//-----> Compute FFT on the chunk
	if(CUFFT){
		//---------> FFT
		cufftHandle plan;
		cufftResult error;
		error = cufftPlan1d(&plan, nSamples, CUFFT_C2C, nSpectra);
		if (CUFFT_SUCCESS != error){
			printf("CUFFT error: %d", error);
		}
		
		timer.Start();
		cufftExecC2C(plan, (cufftComplex *)d_input, (cufftComplex *)d_output, CUFFT_FORWARD);
		timer.Stop();
		cuFFT_time += timer.Elapsed();
		
		cufftDestroy(plan);
	}
    
	if(MULTIPLE){
		if (DEBUG) printf("Multiple FFT...: \t\t\t");
		FFT_init();
		FFT_multiple_benchmark(d_input, d_output, nSamples, nSpectra, &FFT_multiple_time);
		if (DEBUG) printf("done in %g ms.\n", FFT_multiple_time);
	}
	
	if(EXTERNAL){
		if (DEBUG) printf("FFT...: \t\t\t\t");
		FFT_init();
		FFT_external_benchmark(d_input, d_output, nSamples, nSpectra, &FFT_external_time);
		if (DEBUG) printf("done in %g ms.\n", FFT_external_time);
	}
	
	//-----> Copy chunk of output data to host
	if (DEBUG) printf("Transferring data to host...: \t\t");
	timer.Start();
	checkCudaErrors(cudaMemcpy( h_output, d_output, output_size*sizeof(float2), cudaMemcpyDeviceToHost));
	timer.Stop();
	transfer_out+=timer.Elapsed();
	if (DEBUG) printf("done in %g ms.\n", timer.Elapsed());


	

	//---------> error check -----
	checkCudaErrors(cudaGetLastError());
	
	//---------> Feeing allocated resources
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_output));
	
	if (DEBUG || WRITE) printf("nSpectra:%d; nSamples:%d cuFFT:%0.3f ms; FFT:%0.3f ms; FFT external:%0.3f ms; FFT multiple:%0.3f ms;\n",nSpectra,nSamples,cuFFT_time, FFT_time, FFT_external_time, FFT_multiple_time);	
	
	if (WRITE){ 
		char str[200];
		sprintf(str,"GPU_CT.dat");
		if (DEBUG) printf("\n Write results into file...\t");
		save_time(str, nSpectra,nSamples, cuFFT_time, FFT_time, FFT_external_time, FFT_multiple_time, FFT_multiple_reuse_time, FFT_multiple_reuse_registers_time, transfer_in, transfer_out);
		if (DEBUG) printf("\t done.\n-------------------------------------\n");
	}

	return(1);
}
