#include "debug.h"
#include "params.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <fftw3.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void Generate_signal(float *signal, int samples){
	float f1,f2,a1,a2;
	f1=1.0/8.0; f2=2.0/8.0; a1=1.0; a2=0.5;
	
	for(int f=0; f<samples; f++){
		signal[f]=a1*sin(2.0*3.141592654*f1*f) + a2*sin(2.0*3.141592654*f2*f + (3.0*3.141592654)/4.0);
	}
}

void Perform_FFT(float2 *spectra, int nChannels){
	int N=nChannels;
	fftwf_plan p;
	
	p = fftwf_plan_dft_1d(N, (fftwf_complex *) spectra, (fftwf_complex *) spectra, FFTW_FORWARD, FFTW_ESTIMATE);
	fftwf_execute(p);
	fftwf_destroy_plan(p);
}

void Compare_data(float2 *GPU_result, float2 *CPU_result, int N, double *cumulative_error, double *mean_error){
	double error;
	double dtemp;
	
	dtemp=0;
	for(int f=0;f<N;f++){
		error=sqrt( (GPU_result[f].x-CPU_result[f].x)*(GPU_result[f].x-CPU_result[f].x) + (GPU_result[f].y-CPU_result[f].y)*(GPU_result[f].y-CPU_result[f].y) );
		dtemp+=error;
	}
	*cumulative_error=dtemp;
	*mean_error=dtemp/N;
}

void FFT_check(float2 *GPU_result, float2 *input_data, int N, double *cumulative_error, double *mean_error){
	float2 *CPU_result;
	CPU_result	= (float2 *)malloc(N*sizeof(float2));
	double error;
	double dtemp;
	
	for(int c=0;c<N;c++){
		CPU_result[c]=input_data[c];
	}//nChannels
	
	Perform_FFT(CPU_result,N);
	
	dtemp=0;
	for(int f=0;f<N;f++){
		error=sqrt( (GPU_result[f].x-CPU_result[f].x)*(GPU_result[f].x-CPU_result[f].x) + (GPU_result[f].y-CPU_result[f].y)*(GPU_result[f].y-CPU_result[f].y) );
		dtemp+=error;
	}
	*cumulative_error=dtemp;
	*mean_error=dtemp/N;
	delete[] CPU_result;
}

void Full_FFT_check(float2 *GPU_result, float2 *input_data, int nSamples, int nSpectra, double *cumulative_error, double *mean_error){
	float2 *CPU_result;
	CPU_result	= (float2 *)malloc(nSamples*sizeof(float2));
	double dtemp=0;
	double error=0;
	for(int bl=0;bl<nSpectra;bl++){
		for(int c=0;c<nSamples;c++){
			CPU_result[c]=input_data[bl*nSamples + c];
		}//nChannels
		
		Perform_FFT(CPU_result, nSamples);
		
		for(int f=0;f<nSamples;f++){	
			error=sqrt( (GPU_result[bl*nSamples + f].x-CPU_result[f].x)*(GPU_result[bl*nSamples + f].x-CPU_result[f].x) + (GPU_result[bl*nSamples + f].y-CPU_result[f].y)*(GPU_result[bl*nSamples + f].y-CPU_result[f].y) );
			dtemp+=error;
			if(error>1.0e-4) printf("Error! should be: %e,%e but it is %e,%e; at bl:%d f:%d\n",(double) CPU_result[f].x,(double) CPU_result[f].y,(double) GPU_result[bl*nSamples + f].x,(double) GPU_result[bl*nSamples + f].y,bl,f);
		}
		
	} //nSpectra
	*cumulative_error=dtemp;
	*mean_error=dtemp/(nSamples*nSpectra);
	delete[] CPU_result;
}

int GPU_FFT(float2 *input, float2 *output, int nSamples, int nSpectra, int inverse);

//int Max_columns_in_memory_shared(int nTaps, int nDMs);

void Display_data(float2 *input, int samples){
	
	for(int f=0;f<samples;f++){
		printf("%0.4f ",(float) input[f].x);
	}
	printf("\n");
	for(int f=0;f<samples;f++){
		printf("%0.4f ",(float) input[f].y);
	}
	printf("\n");
}

int main(int argc, char* argv[]) {
	
	if (argc!=3) {
		printf("Argument error!\n");
		printf("1) FFT length in power-of-two, i.e. 1024\n");
		printf("2) Number of FFT, i.e. 100000 \n");
        return 1;
    }
	char * pEnd;
	
	int nSamples=strtol(argv[1],&pEnd,10);
	int nSpectra=strtol(argv[2],&pEnd,10);

	//int nColumns=Max_columns_in_memory_shared(nTaps,nDMs);
	
	int input_size=nSpectra*nSamples;
	int output_size=nSpectra*nSamples; if(nSamples<32) output_size=nSpectra*32;
	
	double cumulative_error,mean_error;

	float2 *h_input;
	float2 *h_output;
	float *h_temp;

	if (DEBUG) printf("\nHost memory allocation...\t");
		h_input 	= (float2 *)malloc(input_size*sizeof(float2));
		h_output 	= (float2 *)malloc(output_size*sizeof(float2));
		h_temp	 	= (float *)malloc(output_size*sizeof(float));
	if (DEBUG) printf("done.");

	if (DEBUG) printf("\nHost memory memset...\t\t");
		memset(h_input, 0.0, input_size*sizeof(float2));
		memset(h_output, 0.0, output_size*sizeof(float2));
		memset(h_temp, 0.0, output_size*sizeof(float));
	if (DEBUG) printf("done.");

	if (DEBUG) printf("\nRandom data set...\t\t");	
		Generate_signal(h_temp,nSamples);
		srand(time(NULL));
		for(int f=0;f<nSamples*nSpectra;f++){
			h_input[f].y=rand() / (float)RAND_MAX;
			h_input[f].x=rand() / (float)RAND_MAX;
		}
	if (DEBUG) printf("done.");
	
	GPU_FFT(h_input, h_output, nSamples, nSpectra, 0);

	if (CHECK){
		printf("\nTesting FFT...\n");
		Full_FFT_check(h_output, h_input, nSamples, nSpectra, &cumulative_error, &mean_error);
		printf("Cumulative Error: %e, Mean Error: %e;\n",cumulative_error,mean_error);
	}	
	
	
	delete[] h_input;
	delete[] h_output;
	delete[] h_temp;
	
	cudaDeviceReset();

	if (DEBUG) printf("\nFinished!\n");

	return (0);
}
