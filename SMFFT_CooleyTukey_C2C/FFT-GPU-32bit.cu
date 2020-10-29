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

#include "SM_FFT_parameters.cuh"

int device=0;


__device__ __inline__ float2 Get_W_value(int N, int m){
	float2 ctemp;
	sincosf ( -6.283185308f*fdividef( (float) m, (float) N), &ctemp.y, &ctemp.x);
	return(ctemp);
}

__device__ __inline__ float2 Get_W_value_inverse(int N, int m){
	float2 ctemp;
	sincosf ( 6.283185308f*fdividef( (float) m, (float) N), &ctemp.y, &ctemp.x);
	return(ctemp);
}

__device__ __inline__ float shfl(float *value, int par){
	#if (CUDART_VERSION >= 9000)
		return(__shfl_sync(0xffffffff, (*value), par));
	#else
		return(__shfl((*value), par));
	#endif
}

__device__ __inline__ float shfl_xor(float *value, int par){
	#if (CUDART_VERSION >= 9000)
		return(__shfl_xor_sync(0xffffffff, (*value), par));
	#else
		return(__shfl_xor((*value), par));
	#endif
}

__device__ __inline__ float shfl_down(float *value, int par){
	#if (CUDART_VERSION >= 9000)
		return(__shfl_down_sync(0xffffffff, (*value), par));
	#else
		return(__shfl_down((*value), par));
	#endif
}

__device__ __inline__ void reorder_4_register(float2 *A_DFT_value, float2 *B_DFT_value, float2 *C_DFT_value, float2 *D_DFT_value){
	float2 Af2temp, Bf2temp, Cf2temp, Df2temp;
	unsigned int target = (( (unsigned int) __brev((threadIdx.x&3)) )>>(30)) + 4*(threadIdx.x>>2);
	Af2temp.x = shfl(&(A_DFT_value->x), target);
	Af2temp.y = shfl(&(A_DFT_value->y), target);
	Bf2temp.x = shfl(&(B_DFT_value->x), target);
	Bf2temp.y = shfl(&(B_DFT_value->y), target);
	Cf2temp.x = shfl(&(C_DFT_value->x), target);
	Cf2temp.y = shfl(&(C_DFT_value->y), target);
	Df2temp.x = shfl(&(D_DFT_value->x), target);
	Df2temp.y = shfl(&(D_DFT_value->y), target);
	__syncwarp();
	(*A_DFT_value) = Af2temp;
	(*B_DFT_value) = Bf2temp;
	(*C_DFT_value) = Cf2temp;
	(*D_DFT_value) = Df2temp;
}

__device__ __inline__ void reorder_8_register(float2 *A_DFT_value, float2 *B_DFT_value, float2 *C_DFT_value, float2 *D_DFT_value, int *local_id){
	float2 Af2temp, Bf2temp, Cf2temp, Df2temp;
	unsigned int target = (( (unsigned int) __brev(((*local_id)&7)) )>>(29)) + 8*((*local_id)>>3);
	Af2temp.x = shfl(&(A_DFT_value->x), target);
	Af2temp.y = shfl(&(A_DFT_value->y), target);
	Bf2temp.x = shfl(&(B_DFT_value->x), target);
	Bf2temp.y = shfl(&(B_DFT_value->y), target);
	Cf2temp.x = shfl(&(C_DFT_value->x), target);
	Cf2temp.y = shfl(&(C_DFT_value->y), target);
	Df2temp.x = shfl(&(D_DFT_value->x), target);
	Df2temp.y = shfl(&(D_DFT_value->y), target);
	__syncwarp();
	(*A_DFT_value) = Af2temp;
	(*B_DFT_value) = Bf2temp;
	(*C_DFT_value) = Cf2temp;
	(*D_DFT_value) = Df2temp;
}

__device__ __inline__ void reorder_16_register(float2 *A_DFT_value, float2 *B_DFT_value, float2 *C_DFT_value, float2 *D_DFT_value, int *local_id){
	float2 Af2temp, Bf2temp, Cf2temp, Df2temp;
	unsigned int target = (( (unsigned int) __brev(((*local_id)&15)) )>>(28)) + 16*((*local_id)>>4);
	Af2temp.x = shfl(&(A_DFT_value->x), target);
	Af2temp.y = shfl(&(A_DFT_value->y), target);
	Bf2temp.x = shfl(&(B_DFT_value->x), target);
	Bf2temp.y = shfl(&(B_DFT_value->y), target);
	Cf2temp.x = shfl(&(C_DFT_value->x), target);
	Cf2temp.y = shfl(&(C_DFT_value->y), target);
	Df2temp.x = shfl(&(D_DFT_value->x), target);
	Df2temp.y = shfl(&(D_DFT_value->y), target);
	__syncwarp();
	(*A_DFT_value) = Af2temp;
	(*B_DFT_value) = Bf2temp;
	(*C_DFT_value) = Cf2temp;
	(*D_DFT_value) = Df2temp;
}

__device__ __inline__ void reorder_32_register(float2 *A_DFT_value, float2 *B_DFT_value, float2 *C_DFT_value, float2 *D_DFT_value){
	float2 Af2temp, Bf2temp, Cf2temp, Df2temp;
	unsigned int target = ((unsigned int) __brev( threadIdx.x ))>>(27);
	Af2temp.x = shfl(&(A_DFT_value->x), target);
	Af2temp.y = shfl(&(A_DFT_value->y), target);
	Bf2temp.x = shfl(&(B_DFT_value->x), target);
	Bf2temp.y = shfl(&(B_DFT_value->y), target);
	Cf2temp.x = shfl(&(C_DFT_value->x), target);
	Cf2temp.y = shfl(&(C_DFT_value->y), target);
	Df2temp.x = shfl(&(D_DFT_value->x), target);
	Df2temp.y = shfl(&(D_DFT_value->y), target);
	__syncwarp();
	(*A_DFT_value) = Af2temp;
	(*B_DFT_value) = Bf2temp;
	(*C_DFT_value) = Cf2temp;
	(*D_DFT_value) = Df2temp;
}

template<class const_params>
__device__ __inline__ void reorder_32(float2 *s_input, float2 *A_DFT_value, float2 *B_DFT_value, float2 *C_DFT_value, float2 *D_DFT_value){
	reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
}


template<class const_params>
__device__ __inline__ void reorder_64(float2 *s_input, float2 *A_DFT_value, float2 *B_DFT_value, float2 *C_DFT_value, float2 *D_DFT_value){
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x/const_params::warp;
	
	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
	
	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	__syncthreads();
	unsigned int sm_store_pos = (local_id>>4) + 2*(local_id&15) + warp_id*132;
	s_input[sm_store_pos]          = *A_DFT_value;
	s_input[sm_store_pos + 33]     = *B_DFT_value;
	s_input[66 + sm_store_pos]     = *C_DFT_value;
	s_input[66 + sm_store_pos +33] = *D_DFT_value;
	
	// Read shared memory to get reordered input
	unsigned int sm_read_pos = (local_id&1)*32 + local_id + warp_id*132;
	__syncthreads();
	*A_DFT_value = s_input[sm_read_pos];
	*B_DFT_value = s_input[sm_read_pos + 1];
	*C_DFT_value = s_input[sm_read_pos + 66];
	*D_DFT_value = s_input[sm_read_pos + 66 + 1];
}


template<class const_params>
__device__ __inline__ void reorder_128(float2 *s_input, float2 *A_DFT_value, float2 *B_DFT_value, float2 *C_DFT_value, float2 *D_DFT_value){
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x/const_params::warp;
	
	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
	
	__syncwarp();
	unsigned int sm_store_pos = (local_id>>3) + 4*(local_id&7) + warp_id*132;
	s_input[sm_store_pos]           = *A_DFT_value;
	s_input[sm_store_pos + 33]      = *B_DFT_value;
	s_input[66 + sm_store_pos]      = *C_DFT_value;
	s_input[66 + sm_store_pos + 33] = *D_DFT_value;
	
	// Read shared memory to get reordered input
	__syncwarp();
	unsigned int sm_read_pos = (local_id&3)*32 + local_id + warp_id*132;
	*A_DFT_value = s_input[sm_read_pos];
	*B_DFT_value = s_input[sm_read_pos + 1];
	*C_DFT_value = s_input[sm_read_pos + 2];
	*D_DFT_value = s_input[sm_read_pos + 3];
	
	__syncwarp();
	reorder_4_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
}


template<class const_params>
__device__ __inline__ void reorder_256(float2 *s_input, float2 *A_DFT_value, float2 *B_DFT_value, float2 *C_DFT_value, float2 *D_DFT_value){
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x/const_params::warp;
	
	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
	
	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	__syncthreads();
	unsigned int sm_store_pos = (local_id>>2) + 8*(local_id&3) + warp_id*132;
	s_input[sm_store_pos]           = *A_DFT_value;
	s_input[sm_store_pos + 33]      = *B_DFT_value;
	s_input[66 + sm_store_pos]      = *C_DFT_value;
	s_input[66 + sm_store_pos + 33] = *D_DFT_value;
	
	// Read shared memory to get reordered input
	__syncthreads();
	unsigned int sm_read_pos = (local_id&7)*32 + local_id;
	*A_DFT_value = s_input[sm_read_pos + warp_id*4 + 0];
	*B_DFT_value = s_input[sm_read_pos + warp_id*4 + 1];
	*C_DFT_value = s_input[sm_read_pos + warp_id*4 + 2];
	*D_DFT_value = s_input[sm_read_pos + warp_id*4 + 3];
	
	__syncthreads();
	reorder_8_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value, &local_id);
}

template<class const_params>
__device__ __inline__ void reorder_512(float2 *s_input, float2 *A_DFT_value, float2 *B_DFT_value, float2 *C_DFT_value, float2 *D_DFT_value){
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x/const_params::warp;

	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
	
	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	__syncthreads();
	unsigned int sm_store_pos = (local_id>>1) + 16*(local_id&1) + warp_id*132;
	s_input[sm_store_pos]           = *A_DFT_value;
	s_input[sm_store_pos + 33]      = *B_DFT_value;
	s_input[66 + sm_store_pos]      = *C_DFT_value;
	s_input[66 + sm_store_pos + 33] = *D_DFT_value;
	
	// Read shared memory to get reordered input
	unsigned int sm_read_pos = (local_id&15)*32 + local_id  + warp_id*4;
	__syncthreads();
	*A_DFT_value = s_input[sm_read_pos + 0];
	*B_DFT_value = s_input[sm_read_pos + 1];
	*C_DFT_value = s_input[sm_read_pos + 2];
	*D_DFT_value = s_input[sm_read_pos + 3];
	
	__syncthreads();
	reorder_16_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value, &local_id);
}

template<class const_params>
__device__ __inline__ void reorder_1024(float2 *s_input, float2 *A_DFT_value, float2 *B_DFT_value, float2 *C_DFT_value, float2 *D_DFT_value){
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x/const_params::warp;
	
	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
	
	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	__syncthreads();
	unsigned int sm_store_pos = (local_id>>0) + 32*(local_id&0) + warp_id*132;
	s_input[sm_store_pos]           = *A_DFT_value;
	s_input[sm_store_pos + 33]      = *B_DFT_value;
	s_input[66 + sm_store_pos]      = *C_DFT_value;
	s_input[66 + sm_store_pos + 33] = *D_DFT_value;
	
	// Read shared memory to get reordered input
	unsigned int sm_read_pos = (local_id&31)*32 + local_id  + warp_id*4;
	__syncthreads();
	*A_DFT_value = s_input[sm_read_pos + 0];
	*B_DFT_value = s_input[sm_read_pos + 1];
	*C_DFT_value = s_input[sm_read_pos + 2];
	*D_DFT_value = s_input[sm_read_pos + 3];
	
	__syncthreads();
	reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
}

template<class const_params>
__device__ __inline__ void reorder_2048(float2 *s_input, float2 *A_DFT_value, float2 *B_DFT_value, float2 *C_DFT_value, float2 *D_DFT_value){
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x/const_params::warp;
	
	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
	
	
	__syncthreads();
	//unsigned int sm_store_pos = (local_id>>0) + 32*(local_id&0) + warp_id*132;
	unsigned int sm_store_pos = local_id + warp_id*132;
	s_input[sm_store_pos]      = *A_DFT_value;
	s_input[sm_store_pos + 33] = *B_DFT_value;
	s_input[sm_store_pos + 66] = *C_DFT_value;
	s_input[sm_store_pos + 99] = *D_DFT_value;
	
	// Read shared memory to get reordered input
	__syncthreads();
	//unsigned int sm_read_pos = (local_id&31)*33 + warp_id*2;
	unsigned int sm_read_pos = local_id*33 + warp_id*2;
	*A_DFT_value = s_input[sm_read_pos + 0];
	*B_DFT_value = s_input[sm_read_pos + 1056];
	*C_DFT_value = s_input[sm_read_pos + 1];
	*D_DFT_value = s_input[sm_read_pos + 1056 + 1];
	
	__syncthreads();
	reorder_64<const_params>(s_input, A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
}



template<class const_params>
__device__ __inline__ void reorder_4096(float2 *s_input, float2 *A_DFT_value, float2 *B_DFT_value, float2 *C_DFT_value, float2 *D_DFT_value){
	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x/const_params::warp;
	
	// reorder elements within warp so we can save them in semi-transposed manner into shared memory
	reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
	
	__syncthreads();
	//unsigned int sm_store_pos = (local_id>>0) + 32*(local_id&0) + warp_id*132;
	unsigned int sm_store_pos = local_id + warp_id*132;
	s_input[sm_store_pos]      = *A_DFT_value;
	s_input[sm_store_pos + 33] = *B_DFT_value;
	s_input[sm_store_pos + 66] = *C_DFT_value;
	s_input[sm_store_pos + 99] = *D_DFT_value;
	
	// Read shared memory to get reordered input
	__syncthreads();
	//unsigned int sm_read_pos = (local_id&31)*33 + warp_id*2;
	unsigned int sm_read_pos = local_id*33 + warp_id;
	*A_DFT_value = s_input[sm_read_pos + 0];
	*B_DFT_value = s_input[sm_read_pos + 1056];
	*C_DFT_value = s_input[sm_read_pos + 2112];
	*D_DFT_value = s_input[sm_read_pos + 3168];
	
	__syncthreads();
	reorder_128<const_params>(s_input, A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
}




template<class const_params>
__device__ void do_SMFFT_CT_DIT(float2 *s_input){
	float2 A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value;
	float2 W;
	float2 Aftemp, Bftemp, Cftemp, Dftemp;

	int j, m_param;
	int parity, itemp;
	int A_read_index, B_read_index, C_read_index, D_read_index;
	int PoT, PoTp1, q;

	int local_id = threadIdx.x & (const_params::warp - 1);
	int warp_id = threadIdx.x/const_params::warp;
	A_DFT_value = s_input[local_id + (warp_id<<2)*const_params::warp];
	B_DFT_value = s_input[local_id + (warp_id<<2)*const_params::warp + const_params::warp];
	C_DFT_value = s_input[local_id + (warp_id<<2)*const_params::warp + 2*const_params::warp];
	D_DFT_value = s_input[local_id + (warp_id<<2)*const_params::warp + 3*const_params::warp];

	if(const_params::fft_reorder){
		if(const_params::fft_exp==5)       reorder_32<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
		else if(const_params::fft_exp==6)  reorder_64<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
		else if(const_params::fft_exp==7)  reorder_128<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
		else if(const_params::fft_exp==8)  reorder_256<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
		else if(const_params::fft_exp==9)  reorder_512<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
		else if(const_params::fft_exp==10) reorder_1024<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
		else if(const_params::fft_exp==11) reorder_2048<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
		else if(const_params::fft_exp==12) reorder_4096<const_params>(s_input, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
	}
	
	//----> FFT
	PoT=1;
	PoTp1=2;	

	//--> First iteration
	itemp=local_id&1;
	parity=(1-itemp*2);
	
	A_DFT_value.x = parity*A_DFT_value.x + shfl_xor(&A_DFT_value.x, 1);
	A_DFT_value.y = parity*A_DFT_value.y + shfl_xor(&A_DFT_value.y, 1);
	B_DFT_value.x = parity*B_DFT_value.x + shfl_xor(&B_DFT_value.x, 1);
	B_DFT_value.y = parity*B_DFT_value.y + shfl_xor(&B_DFT_value.y, 1);
	C_DFT_value.x = parity*C_DFT_value.x + shfl_xor(&C_DFT_value.x, 1);
	C_DFT_value.y = parity*C_DFT_value.y + shfl_xor(&C_DFT_value.y, 1);
	D_DFT_value.x = parity*D_DFT_value.x + shfl_xor(&D_DFT_value.x, 1);
	D_DFT_value.y = parity*D_DFT_value.y + shfl_xor(&D_DFT_value.y, 1);
	
	//--> Second through Fifth iteration (no synchronization)
	PoT=2;
	PoTp1=4;
	for(q=1;q<5;q++){
		m_param = (local_id & (PoTp1 - 1));
		itemp   = m_param>>q;
		parity  = ((itemp<<1)-1);
		
		if(const_params::fft_direction) W = Get_W_value_inverse(PoTp1, itemp*m_param);
		else W = Get_W_value(PoTp1, itemp*m_param);
		
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
	
	if(const_params::fft_exp==6){
		__syncthreads();
		q = 5;
		m_param = threadIdx.x & (PoT - 1);
		j = threadIdx.x>>q;
		
		if(const_params::fft_direction) W = Get_W_value_inverse(PoTp1,m_param);
		else W = Get_W_value(PoTp1,m_param);
		
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
	
	for(q=5;q<(const_params::fft_exp-1);q++){
		__syncthreads();
		m_param = threadIdx.x & (PoT - 1);
		j=threadIdx.x>>q;
		
		if(const_params::fft_direction) W = Get_W_value_inverse(PoTp1,m_param);
		else W = Get_W_value(PoTp1,m_param);

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
	if(const_params::fft_exp>6) {
		__syncthreads();
		m_param = threadIdx.x;
		
		if(const_params::fft_direction) W = Get_W_value_inverse(PoTp1,m_param);
		else W = Get_W_value(PoTp1,m_param);
		
		A_read_index = m_param;
		B_read_index = m_param + PoT;
		C_read_index = m_param + (PoT>>1);
		D_read_index = m_param + 3*(PoT>>1);
		
		Aftemp = s_input[A_read_index];
		Bftemp = s_input[B_read_index];
		A_DFT_value.x = Aftemp.x + W.x*Bftemp.x - W.y*Bftemp.y;
		A_DFT_value.y = Aftemp.y + W.x*Bftemp.y + W.y*Bftemp.x;		
		B_DFT_value.x = Aftemp.x - W.x*Bftemp.x + W.y*Bftemp.y;
		B_DFT_value.y = Aftemp.y - W.x*Bftemp.y - W.y*Bftemp.x;
		
		Cftemp = s_input[C_read_index];
		Dftemp = s_input[D_read_index];
		if(const_params::fft_direction){
			C_DFT_value.x = Cftemp.x - W.y*Dftemp.x - W.x*Dftemp.y;
			C_DFT_value.y = Cftemp.y - W.y*Dftemp.y + W.x*Dftemp.x;		
			D_DFT_value.x = Cftemp.x + W.y*Dftemp.x + W.x*Dftemp.y;
			D_DFT_value.y = Cftemp.y + W.y*Dftemp.y - W.x*Dftemp.x;
		}
		else {
			C_DFT_value.x = Cftemp.x + W.y*Dftemp.x + W.x*Dftemp.y;
			C_DFT_value.y = Cftemp.y + W.y*Dftemp.y - W.x*Dftemp.x;		
			D_DFT_value.x = Cftemp.x - W.y*Dftemp.x - W.x*Dftemp.y;
			D_DFT_value.y = Cftemp.y - W.y*Dftemp.y + W.x*Dftemp.x;
		}
		
		s_input[A_read_index]=A_DFT_value;
		s_input[B_read_index]=B_DFT_value;
		s_input[C_read_index]=C_DFT_value;
		s_input[D_read_index]=D_DFT_value;
	}
}

template<class const_params>
__global__ void SMFFT_DIT_external(float2 *d_input, float2* d_output) {
	__shared__ float2 s_input[const_params::fft_sm_required];

	s_input[threadIdx.x]                                           = d_input[threadIdx.x + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_length_quarter]        = d_input[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_quarter];
	s_input[threadIdx.x + const_params::fft_length_half]           = d_input[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_half];
	s_input[threadIdx.x + const_params::fft_length_three_quarters] = d_input[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_three_quarters];
	
	__syncthreads();
	do_SMFFT_CT_DIT<const_params>(s_input);
	
	__syncthreads();
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length]                                           = s_input[threadIdx.x];
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_quarter]        = s_input[threadIdx.x + const_params::fft_length_quarter];
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_half]           = s_input[threadIdx.x + const_params::fft_length_half];
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_three_quarters] = s_input[threadIdx.x + const_params::fft_length_three_quarters];
}

template<class const_params>
__global__ void SMFFT_DIT_multiple(float2 *d_input, float2* d_output) {
	__shared__ float2 s_input[const_params::fft_sm_required];
	
	s_input[threadIdx.x]                                           = d_input[threadIdx.x + blockIdx.x*const_params::fft_length];
	s_input[threadIdx.x + const_params::fft_length_quarter]        = d_input[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_quarter];
	s_input[threadIdx.x + const_params::fft_length_half]           = d_input[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_half];
	s_input[threadIdx.x + const_params::fft_length_three_quarters] = d_input[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_three_quarters];
	
	__syncthreads();
	for(int f=0;f<NREUSES;f++){
		do_SMFFT_CT_DIT<const_params>(s_input);
	}
	__syncthreads();
	
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length]                                           = s_input[threadIdx.x];
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_quarter]        = s_input[threadIdx.x + const_params::fft_length_quarter];
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_half]           = s_input[threadIdx.x + const_params::fft_length_half];
	d_output[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_three_quarters] = s_input[threadIdx.x + const_params::fft_length_three_quarters];
}

//---------------------------------- Device End -------------------<

void FFT_init(){
	//---------> Specific nVidia stuff
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
}

int FFT_external_benchmark(float2 *d_input, float2 *d_output, int FFT_size, int nFFTs, bool inverse, bool reorder, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	dim3 gridSize(nFFTs, 1, 1);
	dim3 blockSize(FFT_size/4, 1, 1);
	if(FFT_size==32) {
		gridSize.x = nFFTs/4;
		blockSize.x = 32;
	}
	if(FFT_size==64) {
		gridSize.x = nFFTs/2;
		blockSize.x = 32;
	}
	
	//---------> FFT part
	timer.Start();
	switch(FFT_size) {
		case 32:
			if(inverse==false && reorder==true)  SMFFT_DIT_external<FFT_32_forward><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==false && reorder==false) SMFFT_DIT_external<FFT_32_forward_noreorder><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==true)   SMFFT_DIT_external<FFT_32_inverse><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==false)  SMFFT_DIT_external<FFT_32_inverse_noreorder><<<gridSize, blockSize>>>(d_input, d_output);
			break;
		
		case 64:
			if(inverse==false && reorder==true)  SMFFT_DIT_external<FFT_64_forward><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==false && reorder==false) SMFFT_DIT_external<FFT_64_forward_noreorder><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==true)   SMFFT_DIT_external<FFT_64_inverse><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==false)  SMFFT_DIT_external<FFT_64_inverse_noreorder><<<gridSize, blockSize>>>(d_input, d_output);
			break;
			
		case 128:
			if(inverse==false && reorder==true)  SMFFT_DIT_external<FFT_128_forward><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==false && reorder==false) SMFFT_DIT_external<FFT_128_forward_noreorder><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==true  && reorder==true)   SMFFT_DIT_external<FFT_128_inverse><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==true  && reorder==false)  SMFFT_DIT_external<FFT_128_inverse_noreorder><<<gridSize, blockSize>>>(d_input, d_output);
			break;
		
		case 256:
			if(inverse==false && reorder==true)  SMFFT_DIT_external<FFT_256_forward><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==false && reorder==false) SMFFT_DIT_external<FFT_256_forward_noreorder><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==true)   SMFFT_DIT_external<FFT_256_inverse><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==false)  SMFFT_DIT_external<FFT_256_inverse_noreorder><<<gridSize, blockSize>>>(d_input, d_output);
			break;
			
		case 512:
			if(inverse==false && reorder==true)  SMFFT_DIT_external<FFT_512_forward><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==false && reorder==false) SMFFT_DIT_external<FFT_512_forward_noreorder><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==true)   SMFFT_DIT_external<FFT_512_inverse><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==false)  SMFFT_DIT_external<FFT_512_inverse_noreorder><<<gridSize, blockSize>>>(d_input, d_output);
			break;
		
		case 1024:
			if(inverse==false && reorder==true)  SMFFT_DIT_external<FFT_1024_forward><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==false && reorder==false) SMFFT_DIT_external<FFT_1024_forward_noreorder><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==true)   SMFFT_DIT_external<FFT_1024_inverse><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==false)  SMFFT_DIT_external<FFT_1024_inverse_noreorder><<<gridSize, blockSize>>>(d_input, d_output);
			break;

		case 2048:
			if(inverse==false && reorder==true)  SMFFT_DIT_external<FFT_2048_forward><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==false && reorder==false) SMFFT_DIT_external<FFT_2048_forward_noreorder><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==true)   SMFFT_DIT_external<FFT_2048_inverse><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==false)  SMFFT_DIT_external<FFT_2048_inverse_noreorder><<<gridSize, blockSize>>>(d_input, d_output);
			break;
			
		case 4096:
			if(inverse==false && reorder==true)  SMFFT_DIT_external<FFT_4096_forward><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==false && reorder==false) SMFFT_DIT_external<FFT_4096_forward_noreorder><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==true)   SMFFT_DIT_external<FFT_4096_inverse><<<gridSize, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==false)  SMFFT_DIT_external<FFT_4096_inverse_noreorder><<<gridSize, blockSize>>>(d_input, d_output);
			break;
		
		default : 
			printf("Error wrong FFT length!\n");
			break;
	}
	timer.Stop();
	
	*FFT_time += timer.Elapsed();
	return(0);
}

int FFT_multiple_benchmark(float2 *d_input, float2 *d_output, int FFT_size, int nFFTs, bool inverse, bool reorder, double *FFT_time){
	GpuTimer timer;
	//---------> CUDA block and CUDA grid parameters
	int nBlocks = (int) (nFFTs/NREUSES);
	if(nBlocks == 0) {
		*FFT_time=-1;
		return(1);
	}
	dim3 gridSize_multiple(nBlocks, 1, 1);
	dim3 blockSize(FFT_size/4, 1, 1);
	if(FFT_size==32) {
		gridSize_multiple.x = nFFTs/(4*NREUSES);
		blockSize.x = 32;
	}
	if(FFT_size==64) {
		gridSize_multiple.x = nFFTs/(2*NREUSES);
		blockSize.x = 32;
	}
	
	//---------> FFT part
	timer.Start();
	switch(FFT_size) {
		case 32:
			if(inverse==false && reorder==true)  SMFFT_DIT_multiple<FFT_32_forward><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==false && reorder==false) SMFFT_DIT_multiple<FFT_32_forward_noreorder><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==true)   SMFFT_DIT_multiple<FFT_32_inverse><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==false)  SMFFT_DIT_multiple<FFT_32_inverse_noreorder><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			break;
			
		case 64:
			if(inverse==false && reorder==true)  SMFFT_DIT_multiple<FFT_64_forward><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==false && reorder==false) SMFFT_DIT_multiple<FFT_64_forward_noreorder><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==true)   SMFFT_DIT_multiple<FFT_64_inverse><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==false)  SMFFT_DIT_multiple<FFT_64_inverse_noreorder><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			break;
			
		case 128:
			if(inverse==false && reorder==true)  SMFFT_DIT_multiple<FFT_128_forward><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==false && reorder==false) SMFFT_DIT_multiple<FFT_128_forward_noreorder><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==true)   SMFFT_DIT_multiple<FFT_128_inverse><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==false)  SMFFT_DIT_multiple<FFT_128_inverse_noreorder><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			break;

		case 256:
			if(inverse==false && reorder==true)  SMFFT_DIT_multiple<FFT_256_forward><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==false && reorder==false) SMFFT_DIT_multiple<FFT_256_forward_noreorder><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==true)   SMFFT_DIT_multiple<FFT_256_inverse><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==false)  SMFFT_DIT_multiple<FFT_256_inverse_noreorder><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			break;
			
		case 512:
			if(inverse==false && reorder==true)  SMFFT_DIT_multiple<FFT_512_forward><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==false && reorder==false) SMFFT_DIT_multiple<FFT_512_forward_noreorder><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==true)   SMFFT_DIT_multiple<FFT_512_inverse><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==false)  SMFFT_DIT_multiple<FFT_512_inverse_noreorder><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			break;
		
		case 1024:
			if(inverse==false && reorder==true)  SMFFT_DIT_multiple<FFT_1024_forward><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==false && reorder==false) SMFFT_DIT_multiple<FFT_1024_forward_noreorder><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==true)   SMFFT_DIT_multiple<FFT_1024_inverse><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==false)  SMFFT_DIT_multiple<FFT_1024_inverse_noreorder><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			break;

		case 2048:
			if(inverse==false && reorder==true)  SMFFT_DIT_multiple<FFT_2048_forward><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==false && reorder==false) SMFFT_DIT_multiple<FFT_2048_forward_noreorder><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==true)   SMFFT_DIT_multiple<FFT_2048_inverse><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==false)  SMFFT_DIT_multiple<FFT_2048_inverse_noreorder><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			break;
			
		case 4096:
			if(inverse==false && reorder==true)  SMFFT_DIT_multiple<FFT_4096_forward><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==false && reorder==false) SMFFT_DIT_multiple<FFT_4096_forward_noreorder><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==true)   SMFFT_DIT_multiple<FFT_4096_inverse><<<gridSize_multiple, blockSize>>>(d_input, d_output);
			if(inverse==true && reorder==false)  SMFFT_DIT_multiple<FFT_4096_inverse_noreorder><<<gridSize_multiple, blockSize>>>(d_input, d_output);
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
int GPU_cuFFT(float2 *h_input, float2 *h_output, int FFT_size, int nFFTs, bool inverse, int nRuns, double *single_ex_time){
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
	if(inverse) cufftExecC2C(plan, (cufftComplex *)d_input, (cufftComplex *)d_output, CUFFT_INVERSE);
	else cufftExecC2C(plan, (cufftComplex *)d_input, (cufftComplex *)d_output, CUFFT_FORWARD);
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

int GPU_smFFT_4elements(float2 *h_input, float2 *h_output, int FFT_size, int nFFTs, bool inverse, bool reorder, int nRuns, double *single_ex_time, double *multi_ex_time){
	//---------> Initial nVidia stuff
	int devCount;
	size_t free_mem,total_mem;
	checkCudaErrors(cudaGetDeviceCount(&devCount));
	if(devCount>device) checkCudaErrors(cudaSetDevice(device));
	
	//---------> Checking edge cases
	if(FFT_size==32 && (nFFTs%4)!=0) return(1);
	if(FFT_size==64 && (nFFTs%2)!=0) return(1);
	
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
			FFT_multiple_benchmark(d_input, d_output, FFT_size, nFFTs, inverse, reorder, &total_time_FFT_multiple);
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
			FFT_external_benchmark(d_input, d_output, FFT_size, nFFTs, inverse, reorder, &total_time_FFT_external);
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

