#include "debug.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

float max_error = 1.0e-4;

void Generate_signal(float *signal, int samples){
	float f1,f2,a1,a2;
	f1=1.0/8.0; f2=2.0/8.0; a1=1.0; a2=0.5;
	
	for(int f=0; f<samples; f++){
		signal[f]=a1*sin(2.0*3.141592654*f1*f) + a2*sin(2.0*3.141592654*f2*f + (3.0*3.141592654)/4.0);
	}
}


float get_error(float A, float B){
	float error, div_error=10000, per_error=10000, order=0;
	int power;
	if(A<0) A = -A;
	if(B<0) B = -B;
	
	if (A>B) {
		div_error = A-B;
		if(B>10){
			power = (int) log10(B);
			order = pow(10,power);
			div_error = div_error/order;
		}
	}
	else {
		div_error = B-A;
		if(A>10){
			power = (int) log10(A);
			order = pow(10,power);
			div_error = div_error/order;
		}
	}
	
	if(div_error<per_error) error = div_error;
	else error = per_error;
	return(error);
}

int Compare_data(float2 *cuFFT_result, float2 *smFFT_result, int FFT_size, int nFFTs, double *cumulative_error, double *mean_error){
	double error;
	int nErrors = 0;
	double dtemp;
	int cislo = 0;
	int display = 0;
	
	dtemp=0;
	for(int i=0; i<nFFTs; i++){
		for(int f=0;f<FFT_size;f++){
			int pos = i*FFT_size + f;
			float error_real, error_img;
			error_real = get_error(cuFFT_result[pos].x, smFFT_result[pos].x);
			error_img  = get_error(cuFFT_result[pos].y, smFFT_result[pos].y);
			if(error_real>=error_img) error = error_real; else error = error_img;
			if( (error>max_error || display==1) && cislo<20 ){
				printf("Error=%f; cuFFT=[%f;%f] smFFT=[%f;%f] pos=%d\n", error, cuFFT_result[pos].x, cuFFT_result[pos].y, smFFT_result[pos].x, smFFT_result[pos].y, pos);
				cislo++;
				nErrors++;
			}
			dtemp+=error;
		}
	}
	*cumulative_error = dtemp;
	*mean_error = dtemp/(double) (FFT_size*nFFTs);
	return(nErrors);
}

int GPU_cuFFT(float2 *h_input, float2 *h_output, int FFT_size, int nFFTs, int nRuns, double *single_ex_time);

int GPU_FFT_C2C_Stockham(float2 *h_input, float2 *h_smFFT_output, int FFT_size, int nFFTs, int nRuns, double *single_ex_time, double *multi_ex_time);


int main(int argc, char* argv[]) {
	if (argc!=4) {
		printf("Argument error!\n");
		printf(" 1) FFT length\n");
		printf(" 2) number of FFTs\n");
		printf(" 3) the number of kernel executions\n");
		printf("For example: FFT.exe 1024 100000 20\n");
        return(1);
    }
	char * pEnd;
	
	int FFT_size = strtol(argv[1],&pEnd,10);
	int nFFTs    = strtol(argv[2],&pEnd,10);
	int nRuns = strtol(argv[3],&pEnd,10);
	
	size_t input_size        = nFFTs*FFT_size;
	size_t input_size_bytes  = nFFTs*FFT_size*sizeof(float2);
	size_t output_size       = nFFTs*FFT_size;
	size_t output_size_bytes = nFFTs*FFT_size*sizeof(float2);
	if(DEBUG) printf("FFT size: %d; Number of FFTs: %d; input size: %zu elements = %0.3f MB; output size: %zu elements = %0.3f\n", FFT_size, nFFTs, input_size, input_size_bytes/(1024.0*1024.0), output_size, output_size_bytes/(1024.0*1024.0));
	if(FFT_size<128) { printf("This FFT implementation works for N>=128.\n"); return(1); }

	float2 *h_input;
	float2 *h_smFFT_output;
	float2 *h_cuFFT_output;

	if (DEBUG) printf("\nHost memory allocation...\t");
		h_input        = (float2 *)malloc(input_size_bytes);
		h_smFFT_output = (float2 *)malloc(output_size_bytes);
		h_cuFFT_output = (float2 *)malloc(output_size_bytes);
	if (DEBUG) printf("done.");

	if (DEBUG) printf("\nHost memory memset...\t\t");
		memset(h_input, 0.0, input_size_bytes);
		memset(h_smFFT_output, 0.0, output_size_bytes);
		memset(h_cuFFT_output, 0.0, output_size_bytes);
	if (DEBUG) printf("done.");

	if (DEBUG) printf("\nRandom data set...\t\t");	
		srand(time(NULL));
		for(size_t f=0;f<input_size;f++){
			h_input[f].y=rand() / (float)RAND_MAX;
			h_input[f].x=rand() / (float)RAND_MAX;
		}
	if (DEBUG) printf("done.");
	
	//-----------> cuFFT
	double cuFFT_execution_time;
	GPU_cuFFT(h_input, h_cuFFT_output, FFT_size, nFFTs, nRuns, &cuFFT_execution_time);
	
	//-----------> custom FFT
	double smFFT_execution_time, smFFT_multiple_execution_time;
	GPU_FFT_C2C_Stockham(h_input, h_smFFT_output, FFT_size, nFFTs, nRuns, &smFFT_execution_time, &smFFT_multiple_execution_time);

	#ifdef TESTING
		double cumulative_error,mean_error;
		int nErrors = 0;
		nErrors = Compare_data(h_cuFFT_output, h_smFFT_output, FFT_size, nFFTs, &cumulative_error, &mean_error);
		if(nErrors==0) printf("  FFT test:\033[1;32mPASSED\033[0m\n");
		else printf("  FFT test:\033[1;31mFAILED\033[0m\n");
	#endif
	
	delete[] h_input;
	delete[] h_smFFT_output;
	delete[] h_cuFFT_output;
	
	cudaDeviceReset();

	return (0);
}
