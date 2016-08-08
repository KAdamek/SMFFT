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


int device=0;

/*
__device__ __inline__ float2 Get_W_value(int N, int m){
	float2 ctemp;
	ctemp.x=-cosf( 6.283185f*fdividef( (float) m, (float) N ) - 3.141592654f );
	ctemp.y=sinf( 6.283185f*fdividef( (float) m, (float) N ) - 3.141592654f );
	return(ctemp);
}
*/

__device__ __inline__ float2 Get_W_value(int N, int m){
	float2 ctemp;
	ctemp.x=cosf( -2.0f*3.141592654f*fdividef((int) m, (float) N) );
	ctemp.y=sinf( -2.0f*3.141592654f*fdividef((int) m, (float) N) );
	return(ctemp);
}

__device__ __inline__ float2 Get_W_value_float(float N, float m){
	float2 ctemp;
	ctemp.x=-cosf( 6.283185f*fdividef( m, N) - 3.141592654f );
	ctemp.y=sinf( 6.283185f*fdividef( m, N) - 3.141592654f );
	return(ctemp);
}

__device__ __inline__ float2 Complex_multiplication(float2 A, float2 B){
	float2 C;
	C.x=A.x*B.x - A.y*B.y;
	C.y=A.y*B.x + A.x*B.y;
	return(C);
}

// Arbitrary radix kernel
__device__ void do_AFFT(float2 *s_input, int N, int iter){ // in-place
	float2 DFT_value;
	float2 read_values[RADIX];
	float2 W;
	
	int Tw;
	
	int f,f_read,f_write;
	
	int j, k, r, Lq, Lqm1, rq, N_shift, itemp;
	
	
	N_shift=N/RADIX;
	
	//for(f=0;f<RADIX;f++) read_index[f]=threadIdx.x + f*N_shift;
	
	//-----> FFT
	//--> 
	
	Lq=1;
	Lqm1=0;
	rq=N;
	//if(threadIdx.x==0) printf("Radix is %d\n",RADIX);
	//printf("th:%d; k:%d; Lq:%d; Lqm1:%d; rq:%d; N_shift:%d; \n",threadIdx.x, k, Lq, Lqm1, rq, N_shift);
	//if(threadIdx.x==0) printf("--------------------------------------\n");
	for(r=1;r<=iter;r++){
		Lqm1=Lq;
		Lq=Lq*RADIX;
		rq=rq/RADIX;
		
		//if(threadIdx.x==0) printf("Lqm1:%d; Lq:%d; rq:%d; \n",Lqm1,Lq,rq);
		
		j=threadIdx.x/Lqm1;
		itemp=(int) (threadIdx.x/Lqm1);
		k=(int) (threadIdx.x - itemp*Lqm1);
		
		//printf("th:%d; k:%d; Lq:%d; Lqm1:%d; rq:%d; N_shift:%d; \n",threadIdx.x, k, Lq, Lqm1, rq, N_shift);
		//if(threadIdx.x==0) printf("--------------------------------------\n");
		
		for(f=0;f<RADIX;f++){
			//read_values[f]=s_input[read_index[f]];
			read_values[f]=s_input[threadIdx.x + f*N_shift];
		}

		__syncthreads(); // Must finish reading input values
		
		// first loop in f_write could be out since we do not need to mod the twiddle power
		DFT_value=read_values[0];
		for(f_read=1;f_read<RADIX;f_read++){
			W=Get_W_value(Lq,f_read*k);
			//printf("th:%d; f:%d; k:%d; Lq:%d; \n",threadIdx.x, f_read, f_read*k, Lq);
			DFT_value.x=DFT_value.x + W.x*read_values[f_read].x - W.y*read_values[f_read].y;
			DFT_value.y=DFT_value.y + W.x*read_values[f_read].y + W.y*read_values[f_read].x;
		}
		s_input[j*Lq + k] = DFT_value;
		
		
		for(f_write=1;f_write<RADIX;f_write++){
			DFT_value=read_values[0];
			for(f_read=1;f_read<RADIX;f_read++){
				Tw=f_read*(k+f_write*Lqm1);
				itemp=(int) (Tw/Lq);
				Tw=Tw-itemp*Lq;
				W=Get_W_value(Lq,Tw);
				//printf("th:%d; f_read:%d; f_write:%d; Tw:%d; Lq:%d; \n",threadIdx.x, f_read, f_write, Tw, Lq);
				DFT_value.x=DFT_value.x + W.x*read_values[f_read].x - W.y*read_values[f_read].y;
				DFT_value.y=DFT_value.y + W.x*read_values[f_read].y + W.y*read_values[f_read].x;
			}
			
			s_input[j*Lq + k + f_write*Lqm1] = DFT_value;
		}
		
		__syncthreads(); // Must finish writing output values
	}
	//-------> END
}


__device__ void do_FFT(float2 *s_input, int N, int bits){ // in-place
	float2 DFT_value;
	float2 DFT_value_even, DFT_value_odd, ftemp2, ftemp;
	float2 read_values[RADIX];
	float2 W;
	
	
	int Tw;
	
	int f,f_read,f_write;
	
	int j, k, r, Lq, Lqm1, itemp, A_index, B_index, Nhalf, N_radix;
	// PoT=Lq, PoTm1=Lqm1
	
	//-----> FFT
	//-->
	N_radix=N/RADIX;
	Nhalf=N>>1;
	
	A_index=threadIdx.x;
	B_index=threadIdx.x + Nhalf;
	
	Lq=1;
	Lqm1=0;
	//------------------------------------------------------------
	// First iteration
		Lqm1=Lq;
		Lq=Lq<<1;
		
		j=threadIdx.x;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even.x=ftemp.x + ftemp2.x;
		DFT_value_even.y=ftemp.y + ftemp2.y;
		
		DFT_value_odd.x=ftemp.x - ftemp2.x;
		DFT_value_odd.y=ftemp.y - ftemp2.y;
		
		__syncthreads();
		s_input[j*Lq]=DFT_value_even;
		s_input[j*Lq + Lqm1]=DFT_value_odd;
		__syncthreads();
	// First iteration
	//------------------------------------------------------------
	
	for(r=2;r<=bits;r++){
		Lqm1=Lq;
		Lq=Lq<<1;
		
		j=threadIdx.x>>(r-1);
		k=threadIdx.x & (Lqm1-1);
		
		W=Get_W_value(Lq,k);

		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even.x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even.y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd.x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd.y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;
		
		__syncthreads();
		s_input[j*Lq + k]=DFT_value_even;
		s_input[j*Lq + k + Lqm1]=DFT_value_odd;
		__syncthreads();
	}
	
	
	//N_shift=N !!
	// Last iteration as non-Radix-2 step;
	
	if(RADIX>2){
		Lqm1=Lq;
		Lq=Lq*RADIX;
		
		j=threadIdx.x/Lqm1;
		itemp=(int) (threadIdx.x/Lqm1);
		k=(int) (threadIdx.x - itemp*Lqm1);
		
		if(threadIdx.x<N_radix){
			for(f=0;f<RADIX;f++){
				read_values[f]=s_input[threadIdx.x + f*N_radix];
			}
		}

		__syncthreads(); // Must finish reading input values
		
		if(threadIdx.x<N_radix){
			DFT_value=read_values[0];
			for(f_read=1;f_read<RADIX;f_read++){
				W=Get_W_value(Lq,f_read*k);
				DFT_value.x=DFT_value.x + W.x*read_values[f_read].x - W.y*read_values[f_read].y;
				DFT_value.y=DFT_value.y + W.x*read_values[f_read].y + W.y*read_values[f_read].x;
			}
			s_input[j*Lq + k] = DFT_value;
			
			
			for(f_write=1;f_write<RADIX;f_write++){
				DFT_value=read_values[0];
				for(f_read=1;f_read<RADIX;f_read++){
					Tw=f_read*(k+f_write*Lqm1);
					itemp=(int) (Tw/Lq);
					Tw=Tw-itemp*Lq;
					W=Get_W_value(Lq,Tw);
					DFT_value.x=DFT_value.x + W.x*read_values[f_read].x - W.y*read_values[f_read].y;
					DFT_value.y=DFT_value.y + W.x*read_values[f_read].y + W.y*read_values[f_read].x;
				}
				
				s_input[j*Lq + k + f_write*Lqm1] = DFT_value;
			}
		}
		__syncthreads(); // Must finish writing output values
	}
	//-------> END
}



__global__ void FFT_GPU_external(float2 *d_input, float2* d_output, int N, int bits) {
	extern __shared__ float2 s_input[];
	
	int N_R,f;
	N_R=N/2;
	
	for(f=0;f<2;f++){
		s_input[threadIdx.x + f*N_R]=d_input[threadIdx.x + f*N_R + blockIdx.x*N];
	}
	
	__syncthreads();
	do_FFT(s_input,N,bits);
	__syncthreads();
	
	for(f=0;f<2;f++){
		d_output[threadIdx.x + f*N_R + blockIdx.x*N]=s_input[threadIdx.x + f*N_R];
	}
}

__global__ void FFT_GPU_multiple(float2 *d_input, float2* d_output, int N, int bits) {
	extern __shared__ float2 s_input[];

	int N_R,f;
	N_R=N/2;
	
	for(f=0;f<2;f++){
		s_input[threadIdx.x + f*N_R]=d_input[threadIdx.x + f*N_R + blockIdx.x*N];
	}
	
	__syncthreads();
	for(int f=0;f<100;f++){
		do_FFT(s_input,N,bits);
	}
	
	__syncthreads();
	for(f=0;f<2;f++){
		d_output[threadIdx.x + f*N_R + blockIdx.x*N]=s_input[threadIdx.x + f*N_R];
	}
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
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
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
	FFT_GPU_external<<<gridSize, blockSize,nSamples*8>>>( d_input, d_output, nSamples,round(log(nSamples/RADIX)/log(2)));
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_multiple_benchmark(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize_multiple(1000, 1, 1);	//nCUDAblocks_y goes through spectra
	dim3 blockSize(nSamples/2, 1, 1); 				//nCUDAblocks_x goes through channels
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_multiple<<<gridSize_multiple, blockSize,nSamples*8>>>( d_input, d_output, nSamples,round(log(nSamples/RADIX)/log(2)));
	timer.Stop();
	*FFT_time += timer.Elapsed();
}


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
		sprintf(str,"GPU-FFT-Stockham_Radix-R.dat");
		if (DEBUG) printf("\n Write results into file...\t");
		save_time(str, nSpectra,nSamples, cuFFT_time, FFT_time, FFT_external_time, FFT_multiple_time, FFT_multiple_reuse_time, FFT_multiple_reuse_registers_time, transfer_in, transfer_out);
		if (DEBUG) printf("\t done.\n-------------------------------------\n");
	}
	
	return(1);
}