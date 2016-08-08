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

__device__ __inline__ float2 Get_W_value(int N, int m){
	float2 ctemp;
	ctemp.x=-cosf( 6.283185f*fdividef( (float) m, (float) N ) - 3.141592654f );
	ctemp.y=sinf( 6.283185f*fdividef( (float) m, (float) N ) - 3.141592654f );
	return(ctemp);
}

__device__ __inline__ float2 Get_W_value_float(float N, float m){
	float2 ctemp;
	ctemp.x=-cosf( 6.283185f*fdividef( m, N) - 3.141592654f );
	ctemp.y=sinf( 6.283185f*fdividef( m, N) - 3.141592654f );
	return(ctemp);
}


__device__ void do_FFT(float2 *s_input, int N, int bits){ // in-place
	float2 DFT_value_even[4], DFT_value_odd[4], ftemp2, ftemp;
	float2 W;
	
	int r, j[4], k, PoT, PoTm1, A_index, B_index, Nhalf;

	Nhalf=N>>1;
	
	//-----> FFT
	//--> 
	
	PoT=1;
	PoTm1=0;
	
	// --------------------------------------------------------------------------------------------------------
	// First iteration where we do not actually need to calculate the twiddle factors r=1 k0=0;
		PoTm1=PoT;
		PoT=PoT<<1;
		
		j[0]=threadIdx.x;
		j[1]=(threadIdx.x+blockDim.x);
		j[2]=(threadIdx.x+2*blockDim.x);
		j[3]=(threadIdx.x+3*blockDim.x);
		
		W.x=1;
		W.y=0;
		
		// first two elements of this thread
		A_index=j[0]*PoTm1;
		B_index=j[0]*PoTm1 + Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even[0].x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even[0].y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd[0].x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd[0].y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;
		
		// second two elements of the thread
		A_index=j[1]*PoTm1;
		B_index=j[1]*PoTm1 + Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even[1].x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even[1].y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd[1].x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd[1].y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;		

		// third element
		A_index=j[2]*PoTm1;
		B_index=j[2]*PoTm1 + Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even[2].x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even[2].y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd[2].x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd[2].y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;		

		// fourth element
		A_index=j[3]*PoTm1;
		B_index=j[3]*PoTm1 + Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even[3].x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even[3].y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd[3].x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd[3].y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;		

		
		__syncthreads();
		s_input[j[0]*PoT]=DFT_value_even[0];
		s_input[j[0]*PoT + PoTm1]=DFT_value_odd[0];
		s_input[j[1]*PoT]=DFT_value_even[1];
		s_input[j[1]*PoT + PoTm1]=DFT_value_odd[1];
		s_input[j[2]*PoT]=DFT_value_even[2];
		s_input[j[2]*PoT + PoTm1]=DFT_value_odd[2];
		s_input[j[3]*PoT]=DFT_value_even[3];
		s_input[j[3]*PoT + PoTm1]=DFT_value_odd[3];
		__syncthreads();
	// First iteration
	// --------------------------------------------------------------------------------------------------------
	
	
	for(r=2;r<=(bits-2);r++){
		PoTm1=PoT;
		PoT=PoT<<1;
		
		j[0]=threadIdx.x>>(r-1);
		j[1]=(threadIdx.x+blockDim.x)>>(r-1);
		j[2]=(threadIdx.x+2*blockDim.x)>>(r-1);
		j[3]=(threadIdx.x+3*blockDim.x)>>(r-1);
		k=threadIdx.x & (PoTm1-1);
		
		W=Get_W_value(PoT,k);
		
		// first two elements of this thread
		A_index=j[0]*PoTm1+k;
		B_index=j[0]*PoTm1+k+Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even[0].x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even[0].y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd[0].x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd[0].y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;
		
		// second two elements of the thread
		A_index=j[1]*PoTm1+k;
		B_index=j[1]*PoTm1+k+Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even[1].x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even[1].y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd[1].x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd[1].y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;
		
		// second two elements of the thread
		A_index=j[2]*PoTm1+k;
		B_index=j[2]*PoTm1+k+Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even[2].x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even[2].y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd[2].x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd[2].y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;
		
		// second two elements of the thread
		A_index=j[3]*PoTm1+k;
		B_index=j[3]*PoTm1+k+Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even[3].x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even[3].y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd[3].x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd[3].y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;
		
		__syncthreads();
		s_input[j[0]*PoT + k]=DFT_value_even[0];
		s_input[j[0]*PoT + k + PoTm1]=DFT_value_odd[0];
		s_input[j[1]*PoT + k]=DFT_value_even[1];
		s_input[j[1]*PoT + k + PoTm1]=DFT_value_odd[1];
		s_input[j[2]*PoT + k]=DFT_value_even[2];
		s_input[j[2]*PoT + k + PoTm1]=DFT_value_odd[2];
		s_input[j[3]*PoT + k]=DFT_value_even[3];
		s_input[j[3]*PoT + k + PoTm1]=DFT_value_odd[3];
		__syncthreads();
	}

	// --------------------------------------------------------------------------------------------------------
	// Almost last iteration
		PoTm1=PoT;
		PoT=PoT<<1;
		
		j[0]=threadIdx.x>>(r-1);
		j[1]=j[0];
		j[2]=(threadIdx.x+2*blockDim.x)>>(r-1);
		j[3]=j[2];
		k=threadIdx.x & (PoTm1-1);
		
		// first two elements of this thread
		W=Get_W_value(PoT,k);
		A_index=j[0]*PoTm1+k;
		B_index=j[0]*PoTm1+k+Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even[0].x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even[0].y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd[0].x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd[0].y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;
		
		// third two elements of the thread
		A_index=j[2]*PoTm1 + k;
		B_index=j[2]*PoTm1 + k + Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even[2].x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even[2].y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd[2].x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd[2].y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;
		
		// second two elements of the thread
		W=Get_W_value(PoT,k + blockDim.x);
		A_index=j[1]*PoTm1 + k + blockDim.x;
		B_index=j[1]*PoTm1 + k + blockDim.x + Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even[1].x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even[1].y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd[1].x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd[1].y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;
		
		// forth two elements of the thread
		A_index=j[3]*PoTm1 + k + blockDim.x;
		B_index=j[3]*PoTm1 + k + blockDim.x + Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even[3].x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even[3].y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd[3].x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd[3].y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;
		
		__syncthreads();
		s_input[j[0]*PoT + k]=DFT_value_even[0];
		s_input[j[0]*PoT + k + PoTm1]=DFT_value_odd[0];
		
		s_input[j[1]*PoT + k + blockDim.x]=DFT_value_even[1];
		s_input[j[1]*PoT + k + blockDim.x + PoTm1]=DFT_value_odd[1];
		
		s_input[j[2]*PoT + k]=DFT_value_even[2];
		s_input[j[2]*PoT + k + PoTm1]=DFT_value_odd[2];
		
		s_input[j[3]*PoT + k + blockDim.x]=DFT_value_even[3];
		s_input[j[3]*PoT + k + blockDim.x + PoTm1]=DFT_value_odd[3];
		__syncthreads();
	// Almost last iteration
	// --------------------------------------------------------------------------------------------------------
	
	
	// --------------------------------------------------------------------------------------------------------
	// Last iteration
		PoTm1=PoT;
		PoT=PoT<<1;
		
		j[0]=0;
		j[1]=0;
		j[2]=0;
		j[3]=0;
		k=threadIdx.x;
		
		// first two elements of this thread
		W=Get_W_value(PoT,k);
		A_index=j[0]*PoTm1+k;
		B_index=j[0]*PoTm1+k+Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even[0].x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even[0].y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd[0].x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd[0].y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;
		
		// second two elements of the thread
		W=Get_W_value(PoT,k + blockDim.x);
		A_index=j[1]*PoTm1 + k + blockDim.x;
		B_index=j[1]*PoTm1 + k + blockDim.x + Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even[1].x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even[1].y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd[1].x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd[1].y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;
		
		// third two elements of the thread
		W=Get_W_value(PoT,k + 2*blockDim.x);
		A_index=j[2]*PoTm1 + k + 2*blockDim.x;
		B_index=j[2]*PoTm1 + k + 2*blockDim.x + Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even[2].x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even[2].y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd[2].x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd[2].y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;
		
		// forth two elements of the thread
		W=Get_W_value(PoT,k + 3*blockDim.x);
		A_index=j[3]*PoTm1 + k + 3*blockDim.x;
		B_index=j[3]*PoTm1 + k + 3*blockDim.x + Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even[3].x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even[3].y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd[3].x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd[3].y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;
		
		__syncthreads();
		s_input[j[0]*PoT + k]=DFT_value_even[0];
		s_input[j[0]*PoT + k + PoTm1]=DFT_value_odd[0];
		
		s_input[j[1]*PoT + k + blockDim.x]=DFT_value_even[1];
		s_input[j[1]*PoT + k + blockDim.x + PoTm1]=DFT_value_odd[1];
		
		s_input[j[2]*PoT + k + 2*blockDim.x]=DFT_value_even[2];
		s_input[j[2]*PoT + k + 2*blockDim.x + PoTm1]=DFT_value_odd[2];
		
		s_input[j[3]*PoT + k + 3*blockDim.x]=DFT_value_even[3];
		s_input[j[3]*PoT + k + 3*blockDim.x + PoTm1]=DFT_value_odd[3];
		__syncthreads();
	// Last iteration
	// --------------------------------------------------------------------------------------------------------
	
	
	//-------> END
}


__device__ void do_FFT_reuse(float2 *s_input, float2 *s_twidle, int N, int bits){ //float2 twiddle,
	float2 DFT_value_even, DFT_value_odd, ftemp2, ftemp;
	float2 W;
	
	int r, j, k, PoT, PoTm1, A_index, B_index, Nhalf;

	Nhalf=N>>1;
	
	//-----> FFT
	//--> 
	
	PoT=1;
	PoTm1=0;
	for(r=1;r<=bits;r++){
		PoTm1=PoT;
		PoT=PoT<<1;
		
		j=threadIdx.x>>(r-1);
		k=threadIdx.x & (PoTm1-1);
		
		W=s_twidle[(k*PoT)%N];

		A_index=j*PoTm1+k;
		B_index=j*PoTm1+k+Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even.x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even.y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd.x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd.y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;
		
		__syncthreads();
		s_input[j*PoT + k]=DFT_value_even;
		s_input[j*PoT + k + PoTm1]=DFT_value_odd;
		__syncthreads();
	}
	//-------> END
}


__device__ void do_FFT_reuse_registers(float2 *s_input, float2 *r_twiddle, int N){ //float2 twiddle,
	float2 DFT_value_even, DFT_value_odd, ftemp2, ftemp;
	float2 W;
	
	int r, j, k, PoT, PoTm1, A_index, B_index, Nhalf;

	Nhalf=N>>1;
	
	//-----> FFT
	//--> 
	
	PoT=1;
	PoTm1=0;
	for(r=1;r<=NBITS;r++){
		PoTm1=PoT;
		PoT=PoT<<1;
		
		j=threadIdx.x>>(r-1);
		k=threadIdx.x & (PoTm1-1);
		
		W=r_twiddle[r];

		A_index=j*PoTm1+k;
		B_index=j*PoTm1+k+Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		DFT_value_even.x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even.y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd.x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd.y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;
		
		__syncthreads();
		s_input[j*PoT + k]=DFT_value_even;
		s_input[j*PoT + k + PoTm1]=DFT_value_odd;
		__syncthreads();
	}
	//-------> END
}



__global__ void FFT_GPU_external(float2 *d_input, float2* d_output, int N, int bits) {
	extern __shared__ float2 s_input[];
	for(int f=0; f<8; f++){
		s_input[threadIdx.x + f*(N/8)]=d_input[threadIdx.x + f*(N/8) + blockIdx.x*N];
	}
	
	__syncthreads();
	do_FFT(s_input,N,bits);
	
	__syncthreads();
	for(int f=0; f<8; f++){
		d_output[threadIdx.x + f*(N/8) + blockIdx.x*N]=s_input[threadIdx.x + f*(N/8)];
	}
}


__global__ void FFT_GPU_multiple(float2 *d_input, float2* d_output, int N, int bits) {
	extern __shared__ float2 s_input[];
	for(int f=0; f<8; f++){
		s_input[threadIdx.x + f*(N/8)]=d_input[threadIdx.x + f*(N/8) + blockIdx.x*N];
	}
	
	__syncthreads();
	for(int f=0;f<100;f++){
		do_FFT(s_input,N,bits);
	}
	
	__syncthreads();
	for(int f=0; f<8; f++){
		d_output[threadIdx.x + f*(N/8) + blockIdx.x*N]=s_input[threadIdx.x + f*(N/8)];
	}
}


__global__ void FFT_GPU_multiple_reuse(float2 *d_input, float2* d_output, int N, int bits) {
	extern __shared__ float2 s_input_and_twidle[];
	s_input_and_twidle[threadIdx.x]=d_input[threadIdx.x + blockIdx.x*N];
	s_input_and_twidle[threadIdx.x + N/2]=d_input[threadIdx.x + N/2 + blockIdx.x*N];
	
	s_input_and_twidle[threadIdx.x + N]=Get_W_value(N,threadIdx.x);
	s_input_and_twidle[threadIdx.x + N + N/2]=Get_W_value(N,threadIdx.x + N/2);
	
	__syncthreads();
	for(int f=0;f<100;f++){
		do_FFT_reuse(s_input_and_twidle,&s_input_and_twidle[N],N,bits);
	}
	
	__syncthreads();
	d_output[threadIdx.x + blockIdx.x*N]=s_input_and_twidle[threadIdx.x];
	d_output[threadIdx.x + N/2 + blockIdx.x*N]=s_input_and_twidle[threadIdx.x + N/2];
}

__global__ void FFT_GPU_multiple_reuse_register(float2 *d_input, float2* d_output, int N, int bits) {
	extern __shared__ float2 s_input[];
	float2 r_twiddle[NBITS];
	int r, PoT, PoTm1, k;
	
	//---> Loading input data
	s_input[threadIdx.x]=d_input[threadIdx.x + blockIdx.x*N];
	s_input[threadIdx.x + N/2]=d_input[threadIdx.x + N/2 + blockIdx.x*N];
	
	//---> Calculating twiddle factors
	PoT=1;
	PoTm1=0;
	for(r=1;r<=NBITS;r++){
		PoTm1=PoT;
		PoT=PoT<<1;
		k=threadIdx.x & (PoTm1-1);
		
		r_twiddle[r]=Get_W_value(PoT,k);
	}
	
	__syncthreads();
	for(int f=0;f<100;f++){
		do_FFT_reuse_registers(s_input,r_twiddle,N);
	}
	
	__syncthreads();
	d_output[threadIdx.x + blockIdx.x*N]=s_input[threadIdx.x];
	d_output[threadIdx.x + N/2 + blockIdx.x*N]=s_input[threadIdx.x + N/2];
}

__global__ void FFT_GPU(float2 *d_input, float2* d_output, int N, int bits) {
	extern __shared__ float2 s_input[];
	// ----------> Load phase
	s_input[threadIdx.x]=d_input[threadIdx.x + blockIdx.x*N];
	s_input[threadIdx.x + N/2]=d_input[threadIdx.x + N/2 + blockIdx.x*N];
	
	__syncthreads();
	// ----------> FFT
	float2 DFT_value_even, DFT_value_odd, ftemp2, ftemp;
	float2 W;
	
	int r, j, k, PoT, PoTm1, A_index, B_index, Nhalf;

	Nhalf=N>>1;
	
	//-----> FFT
	//--> 
	
	PoT=1;
	PoTm1=0;
	for(r=1;r<=bits;r++){
		PoTm1=PoT;
		PoT=PoT<<1;
		
		j=threadIdx.x>>(r-1);
		k=threadIdx.x & (PoTm1-1);
		
		W=Get_W_value(PoT,k);

		A_index=j*PoTm1+k;
		B_index=j*PoTm1+k+Nhalf;
		
		ftemp2=s_input[B_index];
		ftemp=s_input[A_index];
		
		//printf("thread:%d; j:%d; k:%d; Writes:%d and %d; Reads:%d and %d\n", threadIdx.x, j, k, (j*PoT + k), (j*PoT + k + PoTm1), A_index, B_index);
		
		DFT_value_even.x=ftemp.x + W.x*ftemp2.x - W.y*ftemp2.y;
		DFT_value_even.y=ftemp.y + W.x*ftemp2.y + W.y*ftemp2.x;
		
		DFT_value_odd.x=ftemp.x - W.x*ftemp2.x + W.y*ftemp2.y;
		DFT_value_odd.y=ftemp.y - W.x*ftemp2.y - W.y*ftemp2.x;
		
		__syncthreads();
		s_input[j*PoT + k]=DFT_value_even;
		s_input[j*PoT + k + PoTm1]=DFT_value_odd;
		__syncthreads();
	}
	
	// ----------> Save phase
	d_output[threadIdx.x + blockIdx.x*N]=s_input[threadIdx.x];
	d_output[threadIdx.x + N/2 + blockIdx.x*N]=s_input[threadIdx.x + N/2];
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

void FFT_benchmark(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	const int multiple=1;
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	int nCUDAblocks_x=nSpectra/multiple;
	int nCUDAblocks_y=1; //Head size
	
	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);	//nCUDAblocks_y goes through spectra
	dim3 blockSize(nSamples/2, 1, 1); 				//nCUDAblocks_x goes through channels
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU<<<gridSize, blockSize, nSamples*8>>>( d_input, d_output, nSamples,round(log(nSamples)/log(2))); //8->4
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_external_benchmark(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	int nCUDAblocks_x=nSpectra;
	int nCUDAblocks_y=1; //Head size
	
	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);	//nCUDAblocks_y goes through spectra
	dim3 blockSize(nSamples/8, 1, 1); 				//nCUDAblocks_x goes through channels
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_external<<<gridSize, blockSize,nSamples*8>>>( d_input, d_output, nSamples,round(log(nSamples)/log(2)));
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_multiple_benchmark(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize_multiple(1000, 1, 1);	//nCUDAblocks_y goes through spectra
	dim3 blockSize(nSamples/8, 1, 1); 				//nCUDAblocks_x goes through channels
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_multiple<<<gridSize_multiple, blockSize,nSamples*8>>>( d_input, d_output, nSamples,round(log(nSamples)/log(2)));
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_multiple_reuse_benchmark(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize_multiple(1000, 1, 1);	//nCUDAblocks_y goes through spectra
	dim3 blockSize(nSamples/2, 1, 1); 				//nCUDAblocks_x goes through channels
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_multiple_reuse<<<gridSize_multiple, blockSize,nSamples*8*2>>>( d_input, d_output, nSamples,round(log(nSamples)/log(2)));
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_multiple_reuse_registers_benchmark(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize_multiple(1000, 1, 1);	//nCUDAblocks_y goes through spectra
	dim3 blockSize(nSamples, 1, 1); 				//nCUDAblocks_x goes through channels
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_multiple_reuse_register<<<gridSize_multiple, blockSize,nSamples*8*2>>>( d_input, d_output, nSamples,round(log(nSamples)/log(2)));
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
		sprintf(str,"GPU-FFT-Stockham.dat");
		if (DEBUG) printf("\n Write results into file...\t");
		save_time(str, nSpectra,nSamples, cuFFT_time, FFT_time, FFT_external_time, FFT_multiple_time, FFT_multiple_reuse_time, FFT_multiple_reuse_registers_time, transfer_in, transfer_out);
		if (DEBUG) printf("\t done.\n-------------------------------------\n");
	}
	
	return(1);
}
