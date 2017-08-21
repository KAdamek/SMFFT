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

__device__ __inline__ float2 Get_W_value_inverse(int N, int m){
	float2 ctemp;
	ctemp.x=cosf( 2.0f*3.141592654f*fdividef( (float) m, (float) N) );
	ctemp.y=sinf( 2.0f*3.141592654f*fdividef( (float) m, (float) N) );
	return(ctemp);
}


__device__ void do_FFT(float2 *s_input, int N, int bits){
	float2 A_DFT_value, B_DFT_value;
	float2 W;
	float2 Aftemp, Bftemp;

	int local_id, warp_id;
	int j, m_param;
	int parity, itemp;
	int A_read_index,B_read_index;
	int PoT, PoTp1, q;
	
	local_id = threadIdx.x & (WARP - 1);
	warp_id = threadIdx.x/WARP;

	#ifdef REORDER
	int load_id, i, n;
	load_id = threadIdx.x;
	n=threadIdx.x;
	for(i=1; i<bits; i++) {
		n >>= 1;
		load_id <<= 1;
		load_id |= n & 1;
    }
    load_id &= N-1;
	
	//-----> Scrambling input
	A_DFT_value=s_input[load_id];
	B_DFT_value=s_input[load_id + 1];
	//A_DFT_value=s_input[__brev(threadIdx.x)];
	//B_DFT_value=s_input[__brev(threadIdx.x) + 1];
	__syncthreads();
	s_input[threadIdx.x]     = A_DFT_value;
	s_input[threadIdx.x+N/2] = B_DFT_value;
	__syncthreads();
	#endif
	
	
	//-----> FFT
	//-->
	PoT=1;
	PoTp1=2;	

	//--> First iteration
	itemp=local_id&1;
	parity=(1-itemp*2);
	A_DFT_value=s_input[local_id + warp_id*2*WARP];
	B_DFT_value=s_input[local_id + warp_id*2*WARP + WARP];
	
	__syncthreads();
	
	A_DFT_value.x=parity*A_DFT_value.x + __shfl_xor(A_DFT_value.x,1);
	A_DFT_value.y=parity*A_DFT_value.y + __shfl_xor(A_DFT_value.y,1);
	
	B_DFT_value.x=parity*B_DFT_value.x + __shfl_xor(B_DFT_value.x,1);
	B_DFT_value.y=parity*B_DFT_value.y + __shfl_xor(B_DFT_value.y,1);
	
	
	//--> Second through Fifth iteration (no synchronization)
	PoT=2;   // power of two
	PoTp1=4;
	for(q=1;q<5;q++){
		m_param = (local_id & (PoTp1 - 1)); // twiddle factor index
		// itemp gives 0 for upper half of the butterfly 1 for lower half
		itemp = m_param>>q;
		parity=(itemp*2-1); // parity is -1 or +1 depending on itemp
		// twiddle factor w^0 for upper half w^m_param for lower half
		W = Get_W_value(PoTp1, itemp*m_param);
		
		// first we multiply element held by the thread by a twiddle factor
		Aftemp.x = W.x*A_DFT_value.x - W.y*A_DFT_value.y;
		Aftemp.y = W.x*A_DFT_value.y + W.y*A_DFT_value.x;
		Bftemp.x = W.x*B_DFT_value.x - W.y*B_DFT_value.y;
		Bftemp.y = W.x*B_DFT_value.y + W.y*B_DFT_value.x;

		// then we use shuffle to compute new element of longer FFT
		A_DFT_value.x = Aftemp.x + parity*__shfl_xor(Aftemp.x,PoT);
		A_DFT_value.y = Aftemp.y + parity*__shfl_xor(Aftemp.y,PoT);
		B_DFT_value.x = Bftemp.x + parity*__shfl_xor(Bftemp.x,PoT);
		B_DFT_value.y = Bftemp.y + parity*__shfl_xor(Bftemp.y,PoT);

		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
	
	s_input[local_id + warp_id*2*WARP]=A_DFT_value;
	s_input[local_id + warp_id*2*WARP + WARP]=B_DFT_value;
	
	for(q=5;q<FFT_EXP;q++){
		__syncthreads();
		m_param = threadIdx.x & (PoT - 1);
		j=threadIdx.x>>q;
		W=Get_W_value(PoTp1,m_param);
		//position of element A and B
		A_read_index=j*PoTp1 + m_param;
		B_read_index=j*PoTp1 + m_param + PoT;
		Aftemp = s_input[A_read_index];
		Bftemp = s_input[B_read_index];
		
		// calculation of C_p=A+w^m_param*B
		A_DFT_value.x=Aftemp.x + W.x*Bftemp.x - W.y*Bftemp.y;
		A_DFT_value.y=Aftemp.y + W.x*Bftemp.y + W.y*Bftemp.x;
		// calculation of C_m=A-w^m_param*B
		B_DFT_value.x=Aftemp.x - W.x*Bftemp.x + W.y*Bftemp.y;
		B_DFT_value.y=Aftemp.y - W.x*Bftemp.y - W.y*Bftemp.x;
		
		// Store back in shared memory
		s_input[A_read_index] = A_DFT_value;
		s_input[B_read_index] = B_DFT_value;
		
		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
}


__device__ void do_FFT_2way_2vertical(float2 *s_input_1, float2 *s_input_2){
	float2 A_DFT_value1, B_DFT_value1;
	float2 A_DFT_value2, B_DFT_value2;
	float2 W;
	float2 Aftemp1, Bftemp1;
	float2 Aftemp2, Bftemp2;

	int local_id, warp_id;
	int j, m_param;
	int parity, itemp;
	int A_read_index, B_read_index;
	int PoT, PoTp1, q;
	
	local_id = threadIdx.x & (WARP - 1);
	warp_id = threadIdx.x/WARP;

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
	A_DFT_value1=s_input_1[load_id];
	B_DFT_value1=s_input_1[load_id + 1];
	A_DFT_value2=s_input_2[load_id];
	B_DFT_value2=s_input_2[load_id + 1];
	__syncthreads();
	s_input_1[threadIdx.x]              = A_DFT_value1;
	s_input_1[threadIdx.x+FFT_LENGTH/2] = B_DFT_value1;
	s_input_2[threadIdx.x]              = A_DFT_value2;
	s_input_2[threadIdx.x+FFT_LENGTH/2] = B_DFT_value2;
	__syncthreads();
	#endif
	
	
	//-----> FFT
	//-->
	PoT=1;
	PoTp1=2;	

	//--> First iteration
	itemp=local_id&1;
	parity=(1-itemp*2);
	A_DFT_value1=s_input_1[local_id + (warp_id<<1)*WARP];
	B_DFT_value1=s_input_1[local_id + (warp_id<<1)*WARP + WARP];
	A_DFT_value2=s_input_2[local_id + (warp_id<<1)*WARP];
	B_DFT_value2=s_input_2[local_id + (warp_id<<1)*WARP + WARP];
	
	__syncthreads();
	
	A_DFT_value1.x=parity*A_DFT_value1.x + __shfl_xor(A_DFT_value1.x,1);
	A_DFT_value1.y=parity*A_DFT_value1.y + __shfl_xor(A_DFT_value1.y,1);
	B_DFT_value1.x=parity*B_DFT_value1.x + __shfl_xor(B_DFT_value1.x,1);
	B_DFT_value1.y=parity*B_DFT_value1.y + __shfl_xor(B_DFT_value1.y,1);
	
	A_DFT_value2.x=parity*A_DFT_value2.x + __shfl_xor(A_DFT_value2.x,1);
	A_DFT_value2.y=parity*A_DFT_value2.y + __shfl_xor(A_DFT_value2.y,1);
	B_DFT_value2.x=parity*B_DFT_value2.x + __shfl_xor(B_DFT_value2.x,1);
	B_DFT_value2.y=parity*B_DFT_value2.y + __shfl_xor(B_DFT_value2.y,1);
	
	
	//--> Second through Fifth iteration (no synchronization)
	PoT=2;
	PoTp1=4;
	for(q=1;q<5;q++){
		m_param = (local_id & (PoTp1 - 1));
		itemp = m_param>>q;
		parity=((itemp<<1)-1);
		W = Get_W_value(PoTp1, itemp*m_param);
		
		Aftemp1.x = W.x*A_DFT_value1.x - W.y*A_DFT_value1.y;
		Aftemp1.y = W.x*A_DFT_value1.y + W.y*A_DFT_value1.x;
		Bftemp1.x = W.x*B_DFT_value1.x - W.y*B_DFT_value1.y;
		Bftemp1.y = W.x*B_DFT_value1.y + W.y*B_DFT_value1.x;
		
		Aftemp2.x = W.x*A_DFT_value2.x - W.y*A_DFT_value2.y;
		Aftemp2.y = W.x*A_DFT_value2.y + W.y*A_DFT_value2.x;
		Bftemp2.x = W.x*B_DFT_value2.x - W.y*B_DFT_value2.y;
		Bftemp2.y = W.x*B_DFT_value2.y + W.y*B_DFT_value2.x;
		
		A_DFT_value1.x = Aftemp1.x + parity*__shfl_xor(Aftemp1.x,PoT);
		A_DFT_value1.y = Aftemp1.y + parity*__shfl_xor(Aftemp1.y,PoT);
		B_DFT_value1.x = Bftemp1.x + parity*__shfl_xor(Bftemp1.x,PoT);
		B_DFT_value1.y = Bftemp1.y + parity*__shfl_xor(Bftemp1.y,PoT);
		
		A_DFT_value2.x = Aftemp2.x + parity*__shfl_xor(Aftemp2.x,PoT);
		A_DFT_value2.y = Aftemp2.y + parity*__shfl_xor(Aftemp2.y,PoT);
		B_DFT_value2.x = Bftemp2.x + parity*__shfl_xor(Bftemp2.x,PoT);
		B_DFT_value2.y = Bftemp2.y + parity*__shfl_xor(Bftemp2.y,PoT);	
		
		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
	
	itemp = local_id + (warp_id<<1)*WARP;
	s_input_1[itemp]          = A_DFT_value1;
	s_input_1[itemp + WARP]   = B_DFT_value1;
	s_input_2[itemp]          = A_DFT_value2;
	s_input_2[itemp + WARP]   = B_DFT_value2;
	
	for(q=5;q<FFT_EXP;q++){
		__syncthreads();
		m_param = threadIdx.x & (PoT - 1);
		j=threadIdx.x>>q;
		
		W=Get_W_value(PoTp1,m_param);

		A_read_index=j*PoTp1 + m_param;
		B_read_index=j*PoTp1 + m_param + PoT;
		
		Aftemp1 = s_input_1[A_read_index];
		Bftemp1 = s_input_1[B_read_index];
		A_DFT_value1.x=Aftemp1.x + W.x*Bftemp1.x - W.y*Bftemp1.y;
		A_DFT_value1.y=Aftemp1.y + W.x*Bftemp1.y + W.y*Bftemp1.x;		
		B_DFT_value1.x=Aftemp1.x - W.x*Bftemp1.x + W.y*Bftemp1.y;
		B_DFT_value1.y=Aftemp1.y - W.x*Bftemp1.y - W.y*Bftemp1.x;
		
		Aftemp2 = s_input_2[A_read_index];
		Bftemp2 = s_input_2[B_read_index];
		A_DFT_value2.x=Aftemp2.x + W.x*Bftemp2.x - W.y*Bftemp2.y;
		A_DFT_value2.y=Aftemp2.y + W.x*Bftemp2.y + W.y*Bftemp2.x;		
		B_DFT_value2.x=Aftemp2.x - W.x*Bftemp2.x + W.y*Bftemp2.y;
		B_DFT_value2.y=Aftemp2.y - W.x*Bftemp2.y - W.y*Bftemp2.x;
		
		s_input_1[A_read_index]=A_DFT_value1;
		s_input_1[B_read_index]=B_DFT_value1;
		s_input_2[A_read_index]=A_DFT_value2;
		s_input_2[B_read_index]=B_DFT_value2;
		
		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
}

__device__ void do_FFT_4way(float2 *s_input, int N, int bits){
	float2 A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value;
	float2 W;
	float2 Aftemp, Bftemp, Cftemp, Dftemp;

	int local_id, warp_id;
	int j, m_param;
	int parity, itemp;
	int A_read_index, B_read_index, C_read_index, D_read_index;
	int PoT, PoTp1, q;
	
	local_id = threadIdx.x & (WARP - 1);
	warp_id = threadIdx.x/WARP;

	#ifdef REORDER
	int A_load_id, B_load_id, i, A_n, B_n;
	A_load_id = threadIdx.x;
	B_load_id = threadIdx.x + N/4;
	A_n=threadIdx.x;
	B_n=threadIdx.x + N/4;
	for(i=1; i<bits; i++) {
		A_n >>= 1;
		B_n >>= 1;
		A_load_id <<= 1;
		A_load_id |= A_n & 1;
		B_load_id <<= 1;
		B_load_id |= B_n & 1;
    }
    A_load_id &= N-1;
	B_load_id &= N-1;
	
	//-----> Scrambling input
	A_DFT_value=s_input[A_load_id];
	B_DFT_value=s_input[A_load_id + 1];
	C_DFT_value=s_input[B_load_id];
	D_DFT_value=s_input[B_load_id + 1];
	__syncthreads();
	s_input[threadIdx.x]         = A_DFT_value;
	s_input[threadIdx.x + N/2]   = B_DFT_value;
	s_input[threadIdx.x + N/4]   = C_DFT_value;
	s_input[threadIdx.x + 3*N/4] = D_DFT_value;
	__syncthreads();
	#endif
	
	
	//-----> FFT
	//-->
	PoT=1;
	PoTp1=2;	

	//--> First iteration
	itemp=local_id&1;
	parity=(1-itemp*2);
	A_DFT_value=s_input[local_id + (warp_id<<2)*WARP];
	B_DFT_value=s_input[local_id + (warp_id<<2)*WARP + WARP];
	C_DFT_value=s_input[local_id + (warp_id<<2)*WARP + 2*WARP];
	D_DFT_value=s_input[local_id + (warp_id<<2)*WARP + 3*WARP];
	
	__syncthreads();
	
	A_DFT_value.x=parity*A_DFT_value.x + __shfl_xor(A_DFT_value.x,1);
	A_DFT_value.y=parity*A_DFT_value.y + __shfl_xor(A_DFT_value.y,1);
	B_DFT_value.x=parity*B_DFT_value.x + __shfl_xor(B_DFT_value.x,1);
	B_DFT_value.y=parity*B_DFT_value.y + __shfl_xor(B_DFT_value.y,1);
	C_DFT_value.x=parity*C_DFT_value.x + __shfl_xor(C_DFT_value.x,1);
	C_DFT_value.y=parity*C_DFT_value.y + __shfl_xor(C_DFT_value.y,1);
	D_DFT_value.x=parity*D_DFT_value.x + __shfl_xor(D_DFT_value.x,1);
	D_DFT_value.y=parity*D_DFT_value.y + __shfl_xor(D_DFT_value.y,1);
	
	
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
		
		A_DFT_value.x = Aftemp.x + parity*__shfl_xor(Aftemp.x,PoT);
		A_DFT_value.y = Aftemp.y + parity*__shfl_xor(Aftemp.y,PoT);
		B_DFT_value.x = Bftemp.x + parity*__shfl_xor(Bftemp.x,PoT);
		B_DFT_value.y = Bftemp.y + parity*__shfl_xor(Bftemp.y,PoT);
		C_DFT_value.x = Cftemp.x + parity*__shfl_xor(Cftemp.x,PoT);
		C_DFT_value.y = Cftemp.y + parity*__shfl_xor(Cftemp.y,PoT);
		D_DFT_value.x = Dftemp.x + parity*__shfl_xor(Dftemp.x,PoT);
		D_DFT_value.y = Dftemp.y + parity*__shfl_xor(Dftemp.y,PoT);	
		
		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
	
	itemp = local_id + (warp_id<<2)*WARP;
	s_input[itemp]          = A_DFT_value;
	s_input[itemp + WARP]   = B_DFT_value;
	s_input[itemp + 2*WARP] = C_DFT_value;
	s_input[itemp + 3*WARP] = D_DFT_value;
	
	for(q=5;q<(FFT_EXP-1);q++){
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

__device__ void do_IFFT_4way(float2 *s_input, int N, int bits){
	float2 A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value;
	float2 W;
	float2 Aftemp, Bftemp, Cftemp, Dftemp;

	int local_id, warp_id;
	int j, m_param;
	int parity, itemp;
	int A_read_index, B_read_index, C_read_index, D_read_index;
	int PoT, PoTp1, q;
	
	local_id = threadIdx.x & (WARP - 1);
	warp_id = threadIdx.x/WARP;

	#ifdef REORDER
	int A_load_id, B_load_id, i, A_n, B_n;
	A_load_id = threadIdx.x;
	B_load_id = threadIdx.x + N/4;
	A_n=threadIdx.x;
	B_n=threadIdx.x + N/4;
	for(i=1; i<bits; i++) {
		A_n >>= 1;
		B_n >>= 1;
		A_load_id <<= 1;
		A_load_id |= A_n & 1;
		B_load_id <<= 1;
		B_load_id |= B_n & 1;
    }
    A_load_id &= N-1;
	B_load_id &= N-1;
	
	//-----> Scrambling input
	A_DFT_value=s_input[A_load_id];
	B_DFT_value=s_input[A_load_id + 1];
	C_DFT_value=s_input[B_load_id];
	D_DFT_value=s_input[B_load_id + 1];
	__syncthreads();
	s_input[threadIdx.x]         = A_DFT_value;
	s_input[threadIdx.x + N/2]   = B_DFT_value;
	s_input[threadIdx.x + N/4]   = C_DFT_value;
	s_input[threadIdx.x + 3*N/4] = D_DFT_value;
	__syncthreads();
	#endif
	
	
	//-----> FFT
	//-->
	PoT=1;
	PoTp1=2;	

	//--> First iteration
	itemp=local_id&1;
	parity=(1-itemp*2);
	A_DFT_value=s_input[local_id + (warp_id<<2)*WARP];
	B_DFT_value=s_input[local_id + (warp_id<<2)*WARP + WARP];
	C_DFT_value=s_input[local_id + (warp_id<<2)*WARP + 2*WARP];
	D_DFT_value=s_input[local_id + (warp_id<<2)*WARP + 3*WARP];
	
	__syncthreads();
	
	A_DFT_value.x=parity*A_DFT_value.x + __shfl_xor(A_DFT_value.x,1);
	A_DFT_value.y=parity*A_DFT_value.y + __shfl_xor(A_DFT_value.y,1);
	B_DFT_value.x=parity*B_DFT_value.x + __shfl_xor(B_DFT_value.x,1);
	B_DFT_value.y=parity*B_DFT_value.y + __shfl_xor(B_DFT_value.y,1);
	C_DFT_value.x=parity*C_DFT_value.x + __shfl_xor(C_DFT_value.x,1);
	C_DFT_value.y=parity*C_DFT_value.y + __shfl_xor(C_DFT_value.y,1);
	D_DFT_value.x=parity*D_DFT_value.x + __shfl_xor(D_DFT_value.x,1);
	D_DFT_value.y=parity*D_DFT_value.y + __shfl_xor(D_DFT_value.y,1);
	
	
	//--> Second through Fifth iteration (no synchronization)
	PoT=2;
	PoTp1=4;
	for(q=1;q<5;q++){
		m_param = (local_id & (PoTp1 - 1));
		itemp = m_param>>q;
		parity=((itemp<<1)-1);
		
		W = Get_W_value_inverse(PoTp1, itemp*m_param);
		
		Aftemp.x = W.x*A_DFT_value.x - W.y*A_DFT_value.y;
		Aftemp.y = W.x*A_DFT_value.y + W.y*A_DFT_value.x;
		Bftemp.x = W.x*B_DFT_value.x - W.y*B_DFT_value.y;
		Bftemp.y = W.x*B_DFT_value.y + W.y*B_DFT_value.x;
		Cftemp.x = W.x*C_DFT_value.x - W.y*C_DFT_value.y;
		Cftemp.y = W.x*C_DFT_value.y + W.y*C_DFT_value.x;
		Dftemp.x = W.x*D_DFT_value.x - W.y*D_DFT_value.y;
		Dftemp.y = W.x*D_DFT_value.y + W.y*D_DFT_value.x;
		
		A_DFT_value.x = Aftemp.x + parity*__shfl_xor(Aftemp.x,PoT);
		A_DFT_value.y = Aftemp.y + parity*__shfl_xor(Aftemp.y,PoT);
		B_DFT_value.x = Bftemp.x + parity*__shfl_xor(Bftemp.x,PoT);
		B_DFT_value.y = Bftemp.y + parity*__shfl_xor(Bftemp.y,PoT);
		C_DFT_value.x = Cftemp.x + parity*__shfl_xor(Cftemp.x,PoT);
		C_DFT_value.y = Cftemp.y + parity*__shfl_xor(Cftemp.y,PoT);
		D_DFT_value.x = Dftemp.x + parity*__shfl_xor(Dftemp.x,PoT);
		D_DFT_value.y = Dftemp.y + parity*__shfl_xor(Dftemp.y,PoT);	
		
		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
	
	itemp = local_id + (warp_id<<2)*WARP;
	s_input[itemp]          = A_DFT_value;
	s_input[itemp + WARP]   = B_DFT_value;
	s_input[itemp + 2*WARP] = C_DFT_value;
	s_input[itemp + 3*WARP] = D_DFT_value;
	
	for(q=5;q<(FFT_EXP-1);q++){
		__syncthreads();
		m_param = threadIdx.x & (PoT - 1);
		j=threadIdx.x>>q;
		
		W=Get_W_value_inverse(PoTp1,m_param);

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
	
	W=Get_W_value_inverse(PoTp1,m_param);
    
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
	C_DFT_value.x=Cftemp.x - W.y*Dftemp.x - W.x*Dftemp.y;
	C_DFT_value.y=Cftemp.y - W.y*Dftemp.y + W.x*Dftemp.x;		
	D_DFT_value.x=Cftemp.x + W.y*Dftemp.x + W.x*Dftemp.y;
	D_DFT_value.y=Cftemp.y + W.y*Dftemp.y - W.x*Dftemp.x;
	
	s_input[A_read_index]=A_DFT_value;
	s_input[B_read_index]=B_DFT_value;
	s_input[C_read_index]=C_DFT_value;
	s_input[D_read_index]=D_DFT_value;	
}

__device__ void do_FFT_4way_2horizontal_1024(float2 *s_input){ //N=1024
	float2 A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value;
	float2 W;
	float2 Aftemp, Bftemp, Cftemp, Dftemp;

	int local_id, warp_id;
	int j, m_param;
	int parity, itemp;
	int A_read_index, B_read_index, C_read_index, D_read_index;
	int PoT, PoTp1, q;
	
	local_id = threadIdx.x & (WARP - 1);
	warp_id = threadIdx.x/WARP;

	#ifdef REORDER
	int load_id, i, n;
	load_id = threadIdx.x;
	n=threadIdx.x;
	for(i=1; i<10; i++){
		n >>= 1;
		load_id <<= 1;
		load_id |= n & 1;
    }
    load_id &= 1023;
	
	//-----> Scrambling input
	A_DFT_value=s_input[load_id];
	B_DFT_value=s_input[load_id + 1];
	C_DFT_value=s_input[load_id + 1024];
	D_DFT_value=s_input[load_id + 1024 + 1];
	__syncthreads();
	s_input[threadIdx.x]        = A_DFT_value;
	s_input[threadIdx.x + 512]  = B_DFT_value;
	s_input[threadIdx.x + 1024] = C_DFT_value;
	s_input[threadIdx.x + 1536] = D_DFT_value;
	__syncthreads();
	#endif
	
	
	//-----> FFT
	//-->
	PoT=1;
	PoTp1=2;	

	//--> First iteration
	itemp=local_id&1;
	parity=(1-itemp*2);
	A_DFT_value=s_input[local_id + (warp_id<<2)*WARP];
	B_DFT_value=s_input[local_id + (warp_id<<2)*WARP + WARP];
	C_DFT_value=s_input[local_id + (warp_id<<2)*WARP + 2*WARP];
	D_DFT_value=s_input[local_id + (warp_id<<2)*WARP + 3*WARP];
	
	__syncthreads();
	
	A_DFT_value.x=parity*A_DFT_value.x + __shfl_xor(A_DFT_value.x,1);
	A_DFT_value.y=parity*A_DFT_value.y + __shfl_xor(A_DFT_value.y,1);
	B_DFT_value.x=parity*B_DFT_value.x + __shfl_xor(B_DFT_value.x,1);
	B_DFT_value.y=parity*B_DFT_value.y + __shfl_xor(B_DFT_value.y,1);
	C_DFT_value.x=parity*C_DFT_value.x + __shfl_xor(C_DFT_value.x,1);
	C_DFT_value.y=parity*C_DFT_value.y + __shfl_xor(C_DFT_value.y,1);
	D_DFT_value.x=parity*D_DFT_value.x + __shfl_xor(D_DFT_value.x,1);
	D_DFT_value.y=parity*D_DFT_value.y + __shfl_xor(D_DFT_value.y,1);
	
	
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
		
		A_DFT_value.x = Aftemp.x + parity*__shfl_xor(Aftemp.x,PoT);
		A_DFT_value.y = Aftemp.y + parity*__shfl_xor(Aftemp.y,PoT);
		B_DFT_value.x = Bftemp.x + parity*__shfl_xor(Bftemp.x,PoT);
		B_DFT_value.y = Bftemp.y + parity*__shfl_xor(Bftemp.y,PoT);
		C_DFT_value.x = Cftemp.x + parity*__shfl_xor(Cftemp.x,PoT);
		C_DFT_value.y = Cftemp.y + parity*__shfl_xor(Cftemp.y,PoT);
		D_DFT_value.x = Dftemp.x + parity*__shfl_xor(Dftemp.x,PoT);
		D_DFT_value.y = Dftemp.y + parity*__shfl_xor(Dftemp.y,PoT);	
		
		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
	
	itemp = local_id + (warp_id<<2)*WARP;
	s_input[itemp]          = A_DFT_value;
	s_input[itemp + WARP]   = B_DFT_value;
	s_input[itemp + 2*WARP] = C_DFT_value;
	s_input[itemp + 3*WARP] = D_DFT_value;
	
	for(q=5;q<10;q++){
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
}

__device__ void do_FFT_4way_2vertical(float2 *s_input1, float2 *s_input2){
	float2 A_DFT_value1, B_DFT_value1, C_DFT_value1, D_DFT_value1;
	float2 A_DFT_value2, B_DFT_value2, C_DFT_value2, D_DFT_value2;
	float2 W;
	float2 Aftemp1, Bftemp1, Cftemp1, Dftemp1;
	float2 Aftemp2, Bftemp2, Cftemp2, Dftemp2;

	int local_id, warp_id;
	int j, m_param;
	int parity, itemp;
	int A_read_index, B_read_index, C_read_index, D_read_index;
	int PoT, PoTp1, q;
	
	local_id = threadIdx.x & (WARP - 1);
	warp_id = threadIdx.x/WARP;

	#ifdef REORDER
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
	A_DFT_value1=s_input1[A_load_id];
	B_DFT_value1=s_input1[A_load_id + 1];
	C_DFT_value1=s_input1[B_load_id];
	D_DFT_value1=s_input1[B_load_id + 1];
	A_DFT_value2=s_input2[A_load_id];
	B_DFT_value2=s_input2[A_load_id + 1];
	C_DFT_value2=s_input2[B_load_id];
	D_DFT_value2=s_input2[B_load_id + 1];
	__syncthreads();
	s_input1[threadIdx.x]         = A_DFT_value1;
	s_input1[threadIdx.x + FFT_LENGTH/2]   = B_DFT_value1;
	s_input1[threadIdx.x + FFT_LENGTH/4]   = C_DFT_value1;
	s_input1[threadIdx.x + 3*FFT_LENGTH/4] = D_DFT_value1;
	s_input2[threadIdx.x]         = A_DFT_value2;
	s_input2[threadIdx.x + FFT_LENGTH/2]   = B_DFT_value2;
	s_input2[threadIdx.x + FFT_LENGTH/4]   = C_DFT_value2;
	s_input2[threadIdx.x + 3*FFT_LENGTH/4] = D_DFT_value2;
	__syncthreads();
	#endif
	
	
	//-----> FFT
	//-->
	PoT=1;
	PoTp1=2;	

	//--> First iteration
	itemp=local_id&1;
	parity=(1-itemp*2);
	A_DFT_value1=s_input1[local_id + (warp_id<<2)*WARP];
	B_DFT_value1=s_input1[local_id + (warp_id<<2)*WARP + WARP];
	C_DFT_value1=s_input1[local_id + (warp_id<<2)*WARP + 2*WARP];
	D_DFT_value1=s_input1[local_id + (warp_id<<2)*WARP + 3*WARP];
	A_DFT_value2=s_input2[local_id + (warp_id<<2)*WARP];
	B_DFT_value2=s_input2[local_id + (warp_id<<2)*WARP + WARP];
	C_DFT_value2=s_input2[local_id + (warp_id<<2)*WARP + 2*WARP];
	D_DFT_value2=s_input2[local_id + (warp_id<<2)*WARP + 3*WARP];
	
	__syncthreads();
	
	A_DFT_value1.x=parity*A_DFT_value1.x + __shfl_xor(A_DFT_value1.x,1);
	A_DFT_value1.y=parity*A_DFT_value1.y + __shfl_xor(A_DFT_value1.y,1);
	B_DFT_value1.x=parity*B_DFT_value1.x + __shfl_xor(B_DFT_value1.x,1);
	B_DFT_value1.y=parity*B_DFT_value1.y + __shfl_xor(B_DFT_value1.y,1);
	C_DFT_value1.x=parity*C_DFT_value1.x + __shfl_xor(C_DFT_value1.x,1);
	C_DFT_value1.y=parity*C_DFT_value1.y + __shfl_xor(C_DFT_value1.y,1);
	D_DFT_value1.x=parity*D_DFT_value1.x + __shfl_xor(D_DFT_value1.x,1);
	D_DFT_value1.y=parity*D_DFT_value1.y + __shfl_xor(D_DFT_value1.y,1);
	
	A_DFT_value2.x=parity*A_DFT_value2.x + __shfl_xor(A_DFT_value2.x,1);
	A_DFT_value2.y=parity*A_DFT_value2.y + __shfl_xor(A_DFT_value2.y,1);
	B_DFT_value2.x=parity*B_DFT_value2.x + __shfl_xor(B_DFT_value2.x,1);
	B_DFT_value2.y=parity*B_DFT_value2.y + __shfl_xor(B_DFT_value2.y,1);
	C_DFT_value2.x=parity*C_DFT_value2.x + __shfl_xor(C_DFT_value2.x,1);
	C_DFT_value2.y=parity*C_DFT_value2.y + __shfl_xor(C_DFT_value2.y,1);
	D_DFT_value2.x=parity*D_DFT_value2.x + __shfl_xor(D_DFT_value2.x,1);
	D_DFT_value2.y=parity*D_DFT_value2.y + __shfl_xor(D_DFT_value2.y,1);
	
	
	//--> Second through Fifth iteration (no synchronization)
	PoT=2;
	PoTp1=4;
	for(q=1;q<5;q++){
		m_param = (local_id & (PoTp1 - 1));
		itemp = m_param>>q;
		parity=((itemp<<1)-1);
		W = Get_W_value(PoTp1, itemp*m_param);
		
		Aftemp1.x = W.x*A_DFT_value1.x - W.y*A_DFT_value1.y;
		Aftemp1.y = W.x*A_DFT_value1.y + W.y*A_DFT_value1.x;
		Bftemp1.x = W.x*B_DFT_value1.x - W.y*B_DFT_value1.y;
		Bftemp1.y = W.x*B_DFT_value1.y + W.y*B_DFT_value1.x;
		Cftemp1.x = W.x*C_DFT_value1.x - W.y*C_DFT_value1.y;
		Cftemp1.y = W.x*C_DFT_value1.y + W.y*C_DFT_value1.x;
		Dftemp1.x = W.x*D_DFT_value1.x - W.y*D_DFT_value1.y;
		Dftemp1.y = W.x*D_DFT_value1.y + W.y*D_DFT_value1.x;
		
		Aftemp2.x = W.x*A_DFT_value2.x - W.y*A_DFT_value2.y;
		Aftemp2.y = W.x*A_DFT_value2.y + W.y*A_DFT_value2.x;
		Bftemp2.x = W.x*B_DFT_value2.x - W.y*B_DFT_value2.y;
		Bftemp2.y = W.x*B_DFT_value2.y + W.y*B_DFT_value2.x;
		Cftemp2.x = W.x*C_DFT_value2.x - W.y*C_DFT_value2.y;
		Cftemp2.y = W.x*C_DFT_value2.y + W.y*C_DFT_value2.x;
		Dftemp2.x = W.x*D_DFT_value2.x - W.y*D_DFT_value2.y;
		Dftemp2.y = W.x*D_DFT_value2.y + W.y*D_DFT_value2.x;
		
		A_DFT_value1.x = Aftemp1.x + parity*__shfl_xor(Aftemp1.x,PoT);
		A_DFT_value1.y = Aftemp1.y + parity*__shfl_xor(Aftemp1.y,PoT);
		B_DFT_value1.x = Bftemp1.x + parity*__shfl_xor(Bftemp1.x,PoT);
		B_DFT_value1.y = Bftemp1.y + parity*__shfl_xor(Bftemp1.y,PoT);
		C_DFT_value1.x = Cftemp1.x + parity*__shfl_xor(Cftemp1.x,PoT);
		C_DFT_value1.y = Cftemp1.y + parity*__shfl_xor(Cftemp1.y,PoT);
		D_DFT_value1.x = Dftemp1.x + parity*__shfl_xor(Dftemp1.x,PoT);
		D_DFT_value1.y = Dftemp1.y + parity*__shfl_xor(Dftemp1.y,PoT);	
		
		A_DFT_value2.x = Aftemp2.x + parity*__shfl_xor(Aftemp2.x,PoT);
		A_DFT_value2.y = Aftemp2.y + parity*__shfl_xor(Aftemp2.y,PoT);
		B_DFT_value2.x = Bftemp2.x + parity*__shfl_xor(Bftemp2.x,PoT);
		B_DFT_value2.y = Bftemp2.y + parity*__shfl_xor(Bftemp2.y,PoT);
		C_DFT_value2.x = Cftemp2.x + parity*__shfl_xor(Cftemp2.x,PoT);
		C_DFT_value2.y = Cftemp2.y + parity*__shfl_xor(Cftemp2.y,PoT);
		D_DFT_value2.x = Dftemp2.x + parity*__shfl_xor(Dftemp2.x,PoT);
		D_DFT_value2.y = Dftemp2.y + parity*__shfl_xor(Dftemp2.y,PoT);	
		
		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
	
	itemp = local_id + (warp_id<<2)*WARP;
	s_input1[itemp]          = A_DFT_value1;
	s_input1[itemp + WARP]   = B_DFT_value1;
	s_input1[itemp + 2*WARP] = C_DFT_value1;
	s_input1[itemp + 3*WARP] = D_DFT_value1;
	
	s_input2[itemp]          = A_DFT_value2;
	s_input2[itemp + WARP]   = B_DFT_value2;
	s_input2[itemp + 2*WARP] = C_DFT_value2;
	s_input2[itemp + 3*WARP] = D_DFT_value2;
	
	for(q=5;q<(FFT_EXP-1);q++){
		__syncthreads();
		m_param = threadIdx.x & (PoT - 1);
		j=threadIdx.x>>q;
		
		W=Get_W_value(PoTp1,m_param);

		A_read_index=j*(PoTp1<<1) + m_param;
		B_read_index=j*(PoTp1<<1) + m_param + PoT;
		C_read_index=j*(PoTp1<<1) + m_param + PoTp1;
		D_read_index=j*(PoTp1<<1) + m_param + 3*PoT;
		
		Aftemp1 = s_input1[A_read_index];
		Bftemp1 = s_input1[B_read_index];
		A_DFT_value1.x=Aftemp1.x + W.x*Bftemp1.x - W.y*Bftemp1.y;
		A_DFT_value1.y=Aftemp1.y + W.x*Bftemp1.y + W.y*Bftemp1.x;		
		B_DFT_value1.x=Aftemp1.x - W.x*Bftemp1.x + W.y*Bftemp1.y;
		B_DFT_value1.y=Aftemp1.y - W.x*Bftemp1.y - W.y*Bftemp1.x;
		
		Aftemp2 = s_input2[A_read_index];
		Bftemp2 = s_input2[B_read_index];
		A_DFT_value2.x=Aftemp2.x + W.x*Bftemp2.x - W.y*Bftemp2.y;
		A_DFT_value2.y=Aftemp2.y + W.x*Bftemp2.y + W.y*Bftemp2.x;		
		B_DFT_value2.x=Aftemp2.x - W.x*Bftemp2.x + W.y*Bftemp2.y;
		B_DFT_value2.y=Aftemp2.y - W.x*Bftemp2.y - W.y*Bftemp2.x;
		
		Cftemp1 = s_input1[C_read_index];
		Dftemp1 = s_input1[D_read_index];
		C_DFT_value1.x=Cftemp1.x + W.x*Dftemp1.x - W.y*Dftemp1.y;
		C_DFT_value1.y=Cftemp1.y + W.x*Dftemp1.y + W.y*Dftemp1.x;		
		D_DFT_value1.x=Cftemp1.x - W.x*Dftemp1.x + W.y*Dftemp1.y;
		D_DFT_value1.y=Cftemp1.y - W.x*Dftemp1.y - W.y*Dftemp1.x;

		Cftemp2 = s_input2[C_read_index];
		Dftemp2 = s_input2[D_read_index];
		C_DFT_value2.x=Cftemp2.x + W.x*Dftemp2.x - W.y*Dftemp2.y;
		C_DFT_value2.y=Cftemp2.y + W.x*Dftemp2.y + W.y*Dftemp2.x;		
		D_DFT_value2.x=Cftemp2.x - W.x*Dftemp2.x + W.y*Dftemp2.y;
		D_DFT_value2.y=Cftemp2.y - W.x*Dftemp2.y - W.y*Dftemp2.x;
		
		s_input1[A_read_index]=A_DFT_value1;
		s_input1[B_read_index]=B_DFT_value1;
		s_input1[C_read_index]=C_DFT_value1;
		s_input1[D_read_index]=D_DFT_value1;
		
		s_input2[A_read_index]=A_DFT_value2;
		s_input2[B_read_index]=B_DFT_value2;
		s_input2[C_read_index]=C_DFT_value2;
		s_input2[D_read_index]=D_DFT_value2;
		
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
	
	Aftemp1 = s_input1[A_read_index];
	Bftemp1 = s_input1[B_read_index];
	A_DFT_value1.x=Aftemp1.x + W.x*Bftemp1.x - W.y*Bftemp1.y;
	A_DFT_value1.y=Aftemp1.y + W.x*Bftemp1.y + W.y*Bftemp1.x;		
	B_DFT_value1.x=Aftemp1.x - W.x*Bftemp1.x + W.y*Bftemp1.y;
	B_DFT_value1.y=Aftemp1.y - W.x*Bftemp1.y - W.y*Bftemp1.x;

	Aftemp2 = s_input2[A_read_index];
	Bftemp2 = s_input2[B_read_index];
	A_DFT_value2.x=Aftemp2.x + W.x*Bftemp2.x - W.y*Bftemp2.y;
	A_DFT_value2.y=Aftemp2.y + W.x*Bftemp2.y + W.y*Bftemp2.x;		
	B_DFT_value2.x=Aftemp2.x - W.x*Bftemp2.x + W.y*Bftemp2.y;
	B_DFT_value2.y=Aftemp2.y - W.x*Bftemp2.y - W.y*Bftemp2.x;	
	
	Cftemp1 = s_input1[C_read_index];
	Dftemp1 = s_input1[D_read_index];
	C_DFT_value1.x=Cftemp1.x + W.y*Dftemp1.x + W.x*Dftemp1.y;
	C_DFT_value1.y=Cftemp1.y + W.y*Dftemp1.y - W.x*Dftemp1.x;		
	D_DFT_value1.x=Cftemp1.x - W.y*Dftemp1.x - W.x*Dftemp1.y;
	D_DFT_value1.y=Cftemp1.y - W.y*Dftemp1.y + W.x*Dftemp1.x;
	
	Cftemp2 = s_input2[C_read_index];
	Dftemp2 = s_input2[D_read_index];
	C_DFT_value2.x=Cftemp2.x + W.y*Dftemp2.x + W.x*Dftemp2.y;
	C_DFT_value2.y=Cftemp2.y + W.y*Dftemp2.y - W.x*Dftemp2.x;		
	D_DFT_value2.x=Cftemp2.x - W.y*Dftemp2.x - W.x*Dftemp2.y;
	D_DFT_value2.y=Cftemp2.y - W.y*Dftemp2.y + W.x*Dftemp2.x;
	
	s_input1[A_read_index]=A_DFT_value1;
	s_input1[B_read_index]=B_DFT_value1;
	s_input1[C_read_index]=C_DFT_value1;
	s_input1[D_read_index]=D_DFT_value1;	
	
	s_input2[A_read_index]=A_DFT_value2;
	s_input2[B_read_index]=B_DFT_value2;
	s_input2[C_read_index]=C_DFT_value2;
	s_input2[D_read_index]=D_DFT_value2;	
}

__device__ void do_FFT_8way(float2 *s_input){
	float2 A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value;
	float2 E_DFT_value, F_DFT_value, G_DFT_value, H_DFT_value;
	float2 W;
	float2 Aftemp, Bftemp, Cftemp, Dftemp;
	float2 Eftemp, Fftemp, Gftemp, Hftemp;

	int local_id, warp_id;
	int j, m_param;
	int parity, itemp;
	int A_read_index, B_read_index, C_read_index, D_read_index;
	int E_read_index, F_read_index, G_read_index, H_read_index;
	int PoT, PoTp1, q;
	
	local_id = threadIdx.x & (WARP - 1);
	warp_id = threadIdx.x/WARP;

	#ifdef REORDER
	int A_load_id, B_load_id, i, A_n, B_n;
	int C_load_id, D_load_id, C_n, D_n;
	A_load_id = threadIdx.x;
	B_load_id = threadIdx.x + FFT_LENGTH/4;
	C_load_id = threadIdx.x + FFT_LENGTH/8;
	D_load_id = threadIdx.x + FFT_LENGTH/4 + FFT_LENGTH/8;
	A_n=threadIdx.x;
	B_n=threadIdx.x + FFT_LENGTH/4;
	C_n=threadIdx.x + FFT_LENGTH/8;
	D_n=threadIdx.x + FFT_LENGTH/4 + FFT_LENGTH/8;
	for(i=1; i<FFT_EXP; i++) {
		A_n >>= 1;
		B_n >>= 1;
		C_n >>= 1;
		D_n >>= 1;
		A_load_id <<= 1;
		A_load_id |= A_n & 1;
		B_load_id <<= 1;
		B_load_id |= B_n & 1;
		C_load_id <<= 1;
		C_load_id |= C_n & 1;
		D_load_id <<= 1;
		D_load_id |= D_n & 1;
    }
    A_load_id &= FFT_LENGTH-1;
	B_load_id &= FFT_LENGTH-1;
	C_load_id &= FFT_LENGTH-1;
	D_load_id &= FFT_LENGTH-1;
	
	//-----> Scrambling input
	A_DFT_value=s_input[A_load_id];
	B_DFT_value=s_input[A_load_id + 1];
	C_DFT_value=s_input[B_load_id];
	D_DFT_value=s_input[B_load_id + 1];
	E_DFT_value=s_input[C_load_id];
	F_DFT_value=s_input[C_load_id + 1];
	G_DFT_value=s_input[D_load_id];
	H_DFT_value=s_input[D_load_id + 1];
	__syncthreads();
	s_input[threadIdx.x]         = A_DFT_value;
	s_input[threadIdx.x + FFT_LENGTH/2]   = B_DFT_value;
	s_input[threadIdx.x + FFT_LENGTH/4]   = C_DFT_value;
	s_input[threadIdx.x + 3*FFT_LENGTH/4] = D_DFT_value;
	
	s_input[threadIdx.x + FFT_LENGTH/8]                = E_DFT_value;
	s_input[threadIdx.x + FFT_LENGTH/8 + FFT_LENGTH/2] = F_DFT_value;
	s_input[threadIdx.x + FFT_LENGTH/4 + FFT_LENGTH/8] = G_DFT_value;
	s_input[threadIdx.x + FFT_LENGTH/4 + FFT_LENGTH/8 + FFT_LENGTH/2] = H_DFT_value;
	__syncthreads();
	#endif
	
	
	//-----> FFT
	//-->
	//--> First iteration
	itemp=local_id&1;
	parity=(1-itemp*2);
	A_DFT_value=s_input[local_id + (warp_id<<3)*WARP];
	B_DFT_value=s_input[local_id + (warp_id<<3)*WARP + WARP];
	C_DFT_value=s_input[local_id + (warp_id<<3)*WARP + 2*WARP];
	D_DFT_value=s_input[local_id + (warp_id<<3)*WARP + 3*WARP];
	E_DFT_value=s_input[local_id + (warp_id<<3)*WARP + 4*WARP];
	F_DFT_value=s_input[local_id + (warp_id<<3)*WARP + 5*WARP];
	G_DFT_value=s_input[local_id + (warp_id<<3)*WARP + 6*WARP];
	H_DFT_value=s_input[local_id + (warp_id<<3)*WARP + 7*WARP];
	
	__syncthreads();
	
	A_DFT_value.x=parity*A_DFT_value.x + __shfl_xor(A_DFT_value.x,1);
	A_DFT_value.y=parity*A_DFT_value.y + __shfl_xor(A_DFT_value.y,1);
	B_DFT_value.x=parity*B_DFT_value.x + __shfl_xor(B_DFT_value.x,1);
	B_DFT_value.y=parity*B_DFT_value.y + __shfl_xor(B_DFT_value.y,1);
	C_DFT_value.x=parity*C_DFT_value.x + __shfl_xor(C_DFT_value.x,1);
	C_DFT_value.y=parity*C_DFT_value.y + __shfl_xor(C_DFT_value.y,1);
	D_DFT_value.x=parity*D_DFT_value.x + __shfl_xor(D_DFT_value.x,1);
	D_DFT_value.y=parity*D_DFT_value.y + __shfl_xor(D_DFT_value.y,1);
	E_DFT_value.x=parity*E_DFT_value.x + __shfl_xor(E_DFT_value.x,1);
	E_DFT_value.y=parity*E_DFT_value.y + __shfl_xor(E_DFT_value.y,1);
	F_DFT_value.x=parity*F_DFT_value.x + __shfl_xor(F_DFT_value.x,1);
	F_DFT_value.y=parity*F_DFT_value.y + __shfl_xor(F_DFT_value.y,1);
	G_DFT_value.x=parity*G_DFT_value.x + __shfl_xor(G_DFT_value.x,1);
	G_DFT_value.y=parity*G_DFT_value.y + __shfl_xor(G_DFT_value.y,1);
	H_DFT_value.x=parity*H_DFT_value.x + __shfl_xor(H_DFT_value.x,1);
	H_DFT_value.y=parity*H_DFT_value.y + __shfl_xor(H_DFT_value.y,1);	
	
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
		Eftemp.x = W.x*E_DFT_value.x - W.y*E_DFT_value.y;
		Eftemp.y = W.x*E_DFT_value.y + W.y*E_DFT_value.x;
		Fftemp.x = W.x*F_DFT_value.x - W.y*F_DFT_value.y;
		Fftemp.y = W.x*F_DFT_value.y + W.y*F_DFT_value.x;
		Gftemp.x = W.x*G_DFT_value.x - W.y*G_DFT_value.y;
		Gftemp.y = W.x*G_DFT_value.y + W.y*G_DFT_value.x;
		Hftemp.x = W.x*H_DFT_value.x - W.y*H_DFT_value.y;
		Hftemp.y = W.x*H_DFT_value.y + W.y*H_DFT_value.x;
		
		A_DFT_value.x = Aftemp.x + parity*__shfl_xor(Aftemp.x,PoT);
		A_DFT_value.y = Aftemp.y + parity*__shfl_xor(Aftemp.y,PoT);
		B_DFT_value.x = Bftemp.x + parity*__shfl_xor(Bftemp.x,PoT);
		B_DFT_value.y = Bftemp.y + parity*__shfl_xor(Bftemp.y,PoT);
		C_DFT_value.x = Cftemp.x + parity*__shfl_xor(Cftemp.x,PoT);
		C_DFT_value.y = Cftemp.y + parity*__shfl_xor(Cftemp.y,PoT);
		D_DFT_value.x = Dftemp.x + parity*__shfl_xor(Dftemp.x,PoT);
		D_DFT_value.y = Dftemp.y + parity*__shfl_xor(Dftemp.y,PoT);	
		E_DFT_value.x = Eftemp.x + parity*__shfl_xor(Eftemp.x,PoT);
		E_DFT_value.y = Eftemp.y + parity*__shfl_xor(Eftemp.y,PoT);
		F_DFT_value.x = Fftemp.x + parity*__shfl_xor(Fftemp.x,PoT);
		F_DFT_value.y = Fftemp.y + parity*__shfl_xor(Fftemp.y,PoT);
		G_DFT_value.x = Gftemp.x + parity*__shfl_xor(Gftemp.x,PoT);
		G_DFT_value.y = Gftemp.y + parity*__shfl_xor(Gftemp.y,PoT);
		H_DFT_value.x = Hftemp.x + parity*__shfl_xor(Hftemp.x,PoT);
		H_DFT_value.y = Hftemp.y + parity*__shfl_xor(Hftemp.y,PoT);	
		
		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
	
	itemp = local_id + (warp_id<<3)*WARP;
	s_input[itemp]          = A_DFT_value;
	s_input[itemp + WARP]   = B_DFT_value;
	s_input[itemp + 2*WARP] = C_DFT_value;
	s_input[itemp + 3*WARP] = D_DFT_value;
	s_input[itemp + 4*WARP] = E_DFT_value;
	s_input[itemp + 5*WARP] = F_DFT_value;
	s_input[itemp + 6*WARP] = G_DFT_value;
	s_input[itemp + 7*WARP] = H_DFT_value;
	
	for(q=5;q<(FFT_EXP-2);q++){
		__syncthreads();
		m_param = threadIdx.x & (PoT - 1);
		j=threadIdx.x>>q;
		
		W=Get_W_value(PoTp1,m_param);

		A_read_index=j*(PoTp1<<2) + m_param;
		B_read_index=j*(PoTp1<<2) + m_param + PoT;
		C_read_index=j*(PoTp1<<2) + m_param + 2*PoT;
		D_read_index=j*(PoTp1<<2) + m_param + 3*PoT;
		E_read_index=j*(PoTp1<<2) + m_param + 4*PoT;
		F_read_index=j*(PoTp1<<2) + m_param + 5*PoT;
		G_read_index=j*(PoTp1<<2) + m_param + 6*PoT;
		H_read_index=j*(PoTp1<<2) + m_param + 7*PoT;
		
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
		
		Eftemp = s_input[E_read_index];
		Fftemp = s_input[F_read_index];
		E_DFT_value.x=Eftemp.x + W.x*Fftemp.x - W.y*Fftemp.y;
		E_DFT_value.y=Eftemp.y + W.x*Fftemp.y + W.y*Fftemp.x;		
		F_DFT_value.x=Eftemp.x - W.x*Fftemp.x + W.y*Fftemp.y;
		F_DFT_value.y=Eftemp.y - W.x*Fftemp.y - W.y*Fftemp.x;
		
		Gftemp = s_input[G_read_index];
		Hftemp = s_input[H_read_index];
		G_DFT_value.x=Gftemp.x + W.x*Hftemp.x - W.y*Hftemp.y;
		G_DFT_value.y=Gftemp.y + W.x*Hftemp.y + W.y*Hftemp.x;		
		H_DFT_value.x=Gftemp.x - W.x*Hftemp.x + W.y*Hftemp.y;
		H_DFT_value.y=Gftemp.y - W.x*Hftemp.y - W.y*Hftemp.x;
		
		s_input[A_read_index]=A_DFT_value;
		s_input[B_read_index]=B_DFT_value;
		s_input[C_read_index]=C_DFT_value;
		s_input[D_read_index]=D_DFT_value;
		s_input[E_read_index]=E_DFT_value;
		s_input[F_read_index]=F_DFT_value;
		s_input[G_read_index]=G_DFT_value;
		s_input[H_read_index]=H_DFT_value;
		
		PoT=PoT<<1;
		PoTp1=PoTp1<<1;
	}
	
	//almost last iteration
	__syncthreads();
	m_param = threadIdx.x;
	W=Get_W_value(PoTp1,m_param);
    
	A_read_index = m_param;
	B_read_index = m_param + 2*(PoT>>1);
	C_read_index = m_param + (PoT>>1);
	D_read_index = m_param + 3*(PoT>>1);
	E_read_index = m_param + 4*(PoT>>1);
	F_read_index = m_param + 6*(PoT>>1);
	G_read_index = m_param + 5*(PoT>>1);
	H_read_index = m_param + 7*(PoT>>1);
	
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
	
	Eftemp = s_input[E_read_index];
	Fftemp = s_input[F_read_index];
	E_DFT_value.x=Eftemp.x + W.x*Fftemp.x - W.y*Fftemp.y;
	E_DFT_value.y=Eftemp.y + W.x*Fftemp.y + W.y*Fftemp.x;		
	F_DFT_value.x=Eftemp.x - W.x*Fftemp.x + W.y*Fftemp.y;
	F_DFT_value.y=Eftemp.y - W.x*Fftemp.y - W.y*Fftemp.x;
	
	Gftemp = s_input[G_read_index];
	Hftemp = s_input[H_read_index];
	G_DFT_value.x=Gftemp.x + W.y*Hftemp.x + W.x*Hftemp.y;
	G_DFT_value.y=Gftemp.y + W.y*Hftemp.y - W.x*Hftemp.x;		
	H_DFT_value.x=Gftemp.x - W.y*Hftemp.x - W.x*Hftemp.y;
	H_DFT_value.y=Gftemp.y - W.y*Hftemp.y + W.x*Hftemp.x;
	
	s_input[A_read_index]=A_DFT_value;
	s_input[B_read_index]=B_DFT_value;
	s_input[C_read_index]=C_DFT_value;
	s_input[D_read_index]=D_DFT_value;
	s_input[E_read_index]=E_DFT_value;
	s_input[F_read_index]=F_DFT_value;
	s_input[G_read_index]=G_DFT_value;
	s_input[H_read_index]=H_DFT_value;	
	
	PoT=PoT<<1;
	PoTp1=PoTp1<<1;
	
	//last iteration
	__syncthreads();
	m_param = threadIdx.x;
	W=Get_W_value(PoTp1,m_param);
	
	A_read_index = m_param;              //+0
	B_read_index = m_param + PoT;        //+512
	C_read_index = m_param + (PoT>>1);   //+256
	D_read_index = m_param + 3*(PoT>>1); //+768
	E_read_index = m_param + (PoT>>2);   //+128
	F_read_index = m_param + 5*(PoT>>2); //+640
	G_read_index = m_param + 3*(PoT>>2); //+384
	H_read_index = m_param + 7*(PoT>>2); //+896
	
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
	
	W=Get_W_value(PoTp1,m_param+(PoT>>2));
	
	Eftemp = s_input[E_read_index];
	Fftemp = s_input[F_read_index];
	E_DFT_value.x=Eftemp.x + W.x*Fftemp.x - W.y*Fftemp.y;
	E_DFT_value.y=Eftemp.y + W.x*Fftemp.y + W.y*Fftemp.x;		
	F_DFT_value.x=Eftemp.x - W.x*Fftemp.x + W.y*Fftemp.y;
	F_DFT_value.y=Eftemp.y - W.x*Fftemp.y - W.y*Fftemp.x;
	
	Gftemp = s_input[G_read_index];
	Hftemp = s_input[H_read_index];
	G_DFT_value.x=Gftemp.x + W.y*Hftemp.x + W.x*Hftemp.y;
	G_DFT_value.y=Gftemp.y + W.y*Hftemp.y - W.x*Hftemp.x;		
	H_DFT_value.x=Gftemp.x - W.y*Hftemp.x - W.x*Hftemp.y;
	H_DFT_value.y=Gftemp.y - W.y*Hftemp.y + W.x*Hftemp.x;
	
	s_input[A_read_index]=A_DFT_value;
	s_input[B_read_index]=B_DFT_value;
	s_input[C_read_index]=C_DFT_value;
	s_input[D_read_index]=D_DFT_value;
	s_input[E_read_index]=E_DFT_value;
	s_input[F_read_index]=F_DFT_value;
	s_input[G_read_index]=G_DFT_value;
	s_input[H_read_index]=H_DFT_value;
	
	__syncthreads();
}
//-----------------------------------------------------------------
//---------------> GPU kernels
__global__ void FFT_GPU_external(float2 *d_input, float2* d_output, int N, int bits) {
	extern __shared__ float2 s_input[];
	s_input[threadIdx.x]=d_input[threadIdx.x + blockIdx.x*N];
	s_input[threadIdx.x + N/2]=d_input[threadIdx.x + blockIdx.x*N + N/2];
	
	__syncthreads();
	do_FFT(s_input,N,bits);
	
	__syncthreads();
	d_output[threadIdx.x + blockIdx.x*N]=s_input[threadIdx.x];
	d_output[threadIdx.x + blockIdx.x*N + N/2]=s_input[threadIdx.x + N/2];
}

__global__ void FFT_GPU_external_2way_2vertical(float2 *d_input, float2* d_output) {
	extern __shared__ float2 s_input[];
	#pragma unroll
	for(int f=0; f<4; f++){	
		s_input[threadIdx.x + f*FFT_LENGTH/2]  = d_input[threadIdx.x + blockIdx.x*2*FFT_LENGTH + f*FFT_LENGTH/2];
	}
	
	__syncthreads();
	do_FFT_2way_2vertical(s_input, &s_input[FFT_LENGTH]);
	__syncthreads();
	
	#pragma unroll
	for(int f=0; f<4; f++){	
		d_output[threadIdx.x + blockIdx.x*2*FFT_LENGTH + f*FFT_LENGTH/2]  = s_input[threadIdx.x + f*FFT_LENGTH/2];
	}
}

__global__ void FFT_GPU_external_4way(float2 *d_input, float2* d_output, int N, int bits) {
	extern __shared__ float2 s_input[];
	s_input[threadIdx.x]=d_input[threadIdx.x + blockIdx.x*N];
	s_input[threadIdx.x + (N>>2)]=d_input[threadIdx.x + blockIdx.x*N + (N>>2)];
	s_input[threadIdx.x + (N>>1)]=d_input[threadIdx.x + blockIdx.x*N + (N>>1)];
	s_input[threadIdx.x + 3*(N>>2)]=d_input[threadIdx.x + blockIdx.x*N + 3*(N>>2)];
	
	__syncthreads();
	do_FFT_4way(s_input,N,bits);
	
	__syncthreads();
	d_output[threadIdx.x + blockIdx.x*N]=s_input[threadIdx.x];
	d_output[threadIdx.x + blockIdx.x*N + (N>>2)]=s_input[threadIdx.x + (N>>2)];
	d_output[threadIdx.x + blockIdx.x*N + (N>>1)]=s_input[threadIdx.x + (N>>1)];
	d_output[threadIdx.x + blockIdx.x*N + 3*(N>>2)]=s_input[threadIdx.x + 3*(N>>2)];
}

__global__ void FFT_GPU_external_8way(float2 *d_input, float2* d_output) {
	extern __shared__ float2 s_input[];
	#pragma unroll
	for(int f=0; f<8; f++){
		s_input[threadIdx.x + f*(FFT_LENGTH>>3)]=d_input[threadIdx.x + blockIdx.x*FFT_LENGTH + f*(FFT_LENGTH>>3)];
	}
	
	__syncthreads();
	do_FFT_8way(s_input);
	
	__syncthreads();
	#pragma unroll
	for(int f=0; f<8; f++){
		d_output[threadIdx.x + blockIdx.x*FFT_LENGTH + f*(FFT_LENGTH>>3)]=s_input[threadIdx.x + f*(FFT_LENGTH>>3)];
	}
}

__global__ void FFT_GPU_external_4way_2horizontal_1024(float2 *d_input, float2* d_output) {
	extern __shared__ float2 s_input[];
	s_input[threadIdx.x]        = d_input[threadIdx.x + blockIdx.x*2*1024];
	s_input[threadIdx.x + 512]  = d_input[threadIdx.x + blockIdx.x*2*1024 + 512];
	s_input[threadIdx.x + 1024] = d_input[threadIdx.x + blockIdx.x*2*1024 + 1024];
	s_input[threadIdx.x + 1536] = d_input[threadIdx.x + blockIdx.x*2*1024 + 1536];
	
	__syncthreads();
	do_FFT_4way_2horizontal_1024(s_input);
	
	__syncthreads();
	d_output[threadIdx.x + blockIdx.x*2*1024]        = s_input[threadIdx.x];
	d_output[threadIdx.x + blockIdx.x*2*1024 + 512]  = s_input[threadIdx.x + 512];
	d_output[threadIdx.x + blockIdx.x*2*1024 + 1024] = s_input[threadIdx.x + 1024];
	d_output[threadIdx.x + blockIdx.x*2*1024 + 1536] = s_input[threadIdx.x + 1536];
}

__global__ void FFT_GPU_external_4way_2vertical(float2 *d_input, float2* d_output) {
	extern __shared__ float2 s_input[];
	#pragma unroll
	for(int f=0; f<8; f++){	
		s_input[threadIdx.x + f*FFT_LENGTH/4]  = d_input[threadIdx.x + blockIdx.x*2*FFT_LENGTH + f*FFT_LENGTH/4];
	}
	
	__syncthreads();
	do_FFT_4way_2vertical(s_input, &s_input[FFT_LENGTH]);
	__syncthreads();
	
	#pragma unroll
	for(int f=0; f<8; f++){	
		d_output[threadIdx.x + blockIdx.x*2*FFT_LENGTH + f*FFT_LENGTH/4]  = s_input[threadIdx.x + f*FFT_LENGTH/4];
	}
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

__global__ void FFT_GPU_multiple_2way_2vertical(float2 *d_input, float2* d_output) {
	extern __shared__ float2 s_input[];
	#pragma unroll
	for(int f=0; f<4; f++){	
		s_input[threadIdx.x + f*FFT_LENGTH/2]  = d_input[threadIdx.x + blockIdx.x*2*FFT_LENGTH + f*FFT_LENGTH/2];
	}
	
	__syncthreads();
	for(int f=0;f<NREUSES;f++){
		do_FFT_2way_2vertical(s_input, &s_input[FFT_LENGTH]);
	}
	__syncthreads();
	
	#pragma unroll
	for(int f=0; f<4; f++){	
		d_output[threadIdx.x + blockIdx.x*2*FFT_LENGTH + f*FFT_LENGTH/2]  = s_input[threadIdx.x + f*FFT_LENGTH/2];
	}
}

__global__ void FFT_GPU_multiple_4way(float2 *d_input, float2* d_output, int N, int bits) {
	extern __shared__ float2 s_input[];
	s_input[threadIdx.x]=d_input[threadIdx.x + blockIdx.x*N];
	s_input[threadIdx.x + (N>>2)]=d_input[threadIdx.x + blockIdx.x*N + (N>>2)];
	s_input[threadIdx.x + (N>>1)]=d_input[threadIdx.x + blockIdx.x*N + (N>>1)];
	s_input[threadIdx.x + 3*(N>>2)]=d_input[threadIdx.x + blockIdx.x*N + 3*(N>>2)];
	
	__syncthreads();
	for(int f=0;f<NREUSES;f++){
		do_FFT_4way(s_input,N,bits);
	}
	__syncthreads();
	
	d_output[threadIdx.x + blockIdx.x*N]=s_input[threadIdx.x];
	d_output[threadIdx.x + blockIdx.x*N + (N>>2)]=s_input[threadIdx.x + (N>>2)];
	d_output[threadIdx.x + blockIdx.x*N + (N>>1)]=s_input[threadIdx.x + (N>>1)];
	d_output[threadIdx.x + blockIdx.x*N + 3*(N>>2)]=s_input[threadIdx.x + 3*(N>>2)];
}

__global__ void FFT_GPU_multiple_8way(float2 *d_input, float2* d_output) {
	extern __shared__ float2 s_input[];
	#pragma unroll
	for(int f=0; f<8; f++){
		s_input[threadIdx.x + f*(FFT_LENGTH>>3)]=d_input[threadIdx.x + blockIdx.x*FFT_LENGTH + f*(FFT_LENGTH>>3)];
	}
	
	__syncthreads();
	for(int f=0;f<NREUSES;f++){
		do_FFT_8way(s_input);
	}
	__syncthreads();

	#pragma unroll
	for(int f=0; f<8; f++){
		d_output[threadIdx.x + blockIdx.x*FFT_LENGTH + f*(FFT_LENGTH>>3)]=s_input[threadIdx.x + f*(FFT_LENGTH>>3)];
	}
}

__global__ void FFT_GPU_multiple_4way_2horizontal_1024(float2 *d_input, float2* d_output) {
	extern __shared__ float2 s_input[];
	s_input[threadIdx.x]        = d_input[threadIdx.x + blockIdx.x*2*1024];
	s_input[threadIdx.x + 512]  = d_input[threadIdx.x + blockIdx.x*2*1024 + 512];
	s_input[threadIdx.x + 1024] = d_input[threadIdx.x + blockIdx.x*2*1024 + 1024];
	s_input[threadIdx.x + 1536] = d_input[threadIdx.x + blockIdx.x*2*1024 + 1536];
	
	__syncthreads();
	for(int f=0;f<NREUSES;f++){
		do_FFT_4way_2horizontal_1024(s_input);
	}
	__syncthreads();
	
	d_output[threadIdx.x + blockIdx.x*2*1024]        = s_input[threadIdx.x];
	d_output[threadIdx.x + blockIdx.x*2*1024 + 512]  = s_input[threadIdx.x + 512];
	d_output[threadIdx.x + blockIdx.x*2*1024 + 1024] = s_input[threadIdx.x + 1024];
	d_output[threadIdx.x + blockIdx.x*2*1024 + 2048] = s_input[threadIdx.x + 1536];
}

__global__ void FFT_GPU_multiple_4way_2vertical(float2 *d_input, float2* d_output) {
	extern __shared__ float2 s_input[];
	#pragma unroll
	for(int f=0; f<8; f++){	
		s_input[threadIdx.x + f*FFT_LENGTH/4]  = d_input[threadIdx.x + blockIdx.x*2*FFT_LENGTH + f*FFT_LENGTH/4];
	}
	
	__syncthreads();
	for(int f=0;f<NREUSES;f++){
		do_FFT_4way_2vertical(s_input, &s_input[FFT_LENGTH]);
	}
	__syncthreads();
	
	#pragma unroll
	for(int f=0; f<8; f++){	
		d_output[threadIdx.x + blockIdx.x*2*FFT_LENGTH + f*FFT_LENGTH/4]  = s_input[threadIdx.x + f*FFT_LENGTH/4];
	}
}
//--------------------------------------<

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
	
	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);	
	dim3 blockSize(nSamples/2, 1, 1); 				
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_external<<<gridSize, blockSize,nSamples*8>>>( d_input, d_output, nSamples,round(log(nSamples)/log(2)));
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_external_benchmark_2way_2vertical(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	int nCUDAblocks_x=nSpectra/2;
	int nCUDAblocks_y=1;
	
	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);
	dim3 blockSize(nSamples/2, 1, 1);
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_external_2way_2vertical<<<gridSize, blockSize,nSamples*2*8>>>( d_input, d_output);
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_external_benchmark_4way(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	int nCUDAblocks_x=nSpectra;
	int nCUDAblocks_y=1; //Head size
	
	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);	
	dim3 blockSize(nSamples/4, 1, 1); 				
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_external_4way<<<gridSize, blockSize,nSamples*8>>>( d_input, d_output, nSamples,round(log(nSamples)/log(2)));
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_external_benchmark_8way(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	int nCUDAblocks_x=nSpectra;
	int nCUDAblocks_y=1; //Head size
	
	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);	
	dim3 blockSize(nSamples/8, 1, 1); 				
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_external_8way<<<gridSize, blockSize,nSamples*8>>>( d_input, d_output );
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_external_benchmark_4way_2horizontal_1024(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	int nCUDAblocks_x=nSpectra/2;
	int nCUDAblocks_y=1;
	
	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);
	dim3 blockSize(512, 1, 1);
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_external_4way_2horizontal_1024<<<gridSize, blockSize, 2048*8>>>( d_input, d_output);
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_external_benchmark_4way_2vertical(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	int nCUDAblocks_x=nSpectra/2;
	int nCUDAblocks_y=1;
	
	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);
	dim3 blockSize(nSamples/4, 1, 1);
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_external_4way_2vertical<<<gridSize, blockSize, nSamples*2*8>>>( d_input, d_output);
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_multiple_benchmark(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize_multiple(NCUDABLOCKS, 1, 1);	
	dim3 blockSize(nSamples/2, 1, 1); 				
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_multiple<<<gridSize_multiple, blockSize,nSamples*8>>>( d_input, d_output, nSamples,round(log(nSamples)/log(2)));
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_multiple_benchmark_2way_2vertical(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize_multiple(NCUDABLOCKS/2, 1, 1);	
	dim3 blockSize(nSamples/2, 1, 1); 				
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_multiple_2way_2vertical<<<gridSize_multiple, blockSize,nSamples*2*8>>>( d_input, d_output);
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_multiple_benchmark_4way(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize_multiple(NCUDABLOCKS, 1, 1);	
	dim3 blockSize(nSamples/4, 1, 1); 				
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_multiple_4way<<<gridSize_multiple, blockSize,nSamples*8>>>( d_input, d_output, nSamples,round(log(nSamples)/log(2)));
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_multiple_benchmark_8way(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize_multiple(NCUDABLOCKS, 1, 1);	
	dim3 blockSize(nSamples/8, 1, 1); 				
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_multiple_8way<<<gridSize_multiple, blockSize,nSamples*8>>>( d_input, d_output );
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_multiple_benchmark_4way_2horizontal_1024(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize_multiple(NCUDABLOCKS/2, 1, 1);	
	dim3 blockSize(512, 1, 1); 				
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_multiple_4way_2horizontal_1024<<<gridSize_multiple, blockSize, 2048*8>>>( d_input, d_output);
	timer.Stop();
	*FFT_time += timer.Elapsed();
}

void FFT_multiple_benchmark_4way_2vertical(float2 *d_input, float2 *d_output, int nSamples, int nSpectra, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize_multiple(NCUDABLOCKS/2, 1, 1);	
	dim3 blockSize(nSamples/4, 1, 1); 				
	
	//---------> FIR filter part
	timer.Start();
	FFT_GPU_multiple_4way_2vertical<<<gridSize_multiple, blockSize, nSamples*2*8>>>( d_input, d_output);
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
	GpuTimer timer;
	
	
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
		
		
		//-------------------------> 8way
		if(MULTIPLE){
			if (DEBUG) printf("\nApplying MULTIPLE FFT 8way...: \t\t");
			FFT_init();
			FFT_multiple_time_total = 0;
			for(int f=0; f<nRuns; f++){
				checkCudaErrors(cudaMemcpy(d_input, &input[nRepeats*output_size], (Sremainder)*nSamples*sizeof(float2), cudaMemcpyHostToDevice));
				FFT_multiple_benchmark_8way(d_input, d_output, nSamples, Sremainder, &FFT_multiple_time_total);
			}
			FFT_multiple_time = FFT_multiple_time_total/nRuns;
			if (DEBUG) printf("done in %g ms.\n", FFT_multiple_time);
		}
		
		if(EXTERNAL){
			if (DEBUG) printf("\nApplying EXTERNAL FFT 8way...: \t\t");
			FFT_init();
			FFT_external_time_total = 0;
			for(int f=0; f<nRuns; f++){
				checkCudaErrors(cudaMemcpy(d_input, &input[nRepeats*output_size], (Sremainder)*nSamples*sizeof(float2), cudaMemcpyHostToDevice));
				FFT_external_benchmark_8way(d_input, d_output, nSamples, Sremainder, &FFT_external_time_total);
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
