#include "debug.h"
#include <iostream>
#include <fstream>
#include <iomanip>

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
	f1=1.1/8.0; f2=3.141592654/8.0; a1=1.0; a2=0.5;
	
	for(int f=0; f<samples; f++){
		signal[f]=a1*sin(2.0*3.141592654*f1*f) + a2*sin(2.0*3.141592654*f2*f + (3.0*3.141592654)/4.0);
	}
}

void Display_data(float *data, int size){
	printf("\n");
	for(int f=0;f<size;f++){
		printf("[%f]; ", data[f]);
	}
	printf("\n");
}

void Display_data(float2 *GPU_result, int size){
	printf("\n");
	for(int f=0;f<size;f++){
		printf("[%f;%f]; ", GPU_result[f].x, GPU_result[f].y);
	}
	printf("\n");
}

void Export(float *data, int size, const char *filename){
	std::ofstream FILEOUT;
	FILEOUT.open(filename);
	for(int f=0; f<size; f++){
		FILEOUT << data[f] << std::endl;
	}
	FILEOUT.close();
}

void Export(float2 *data, int size, const char *filename){
	std::ofstream FILEOUT;
	FILEOUT.open(filename);
	for(int f=0; f<size; f++){
		float sqr = ((data[f].x*data[f].x) + (data[f].y*data[f].y));
		FILEOUT << data[f].x << " " << data[f].y << " " << sqr << std::endl;
	}
	FILEOUT.close();
}

float max(float a, float b){
	if(a>b) return(a);
	else return(b);
}

float get_error(float2 A_f2, float2 B_f2){
	float error, div_error=10000, per_error=10000, order=0;
	int power;
	float A = max(A_f2.x, A_f2.y);
	float B = max(B_f2.x, B_f2.y);
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


int Compare_R2C_output(float2 *kFFT, float2 *cuFFT, int FFT_size, int nFFTs, double *cumulative_error, double *mean_error){
	float error;
	float total_error = 0;
	int nErrors = 0;
	int cislo=0, max_cislo = 0;
	int cuFFT_size = ((FFT_size>>1)+1);
	int kFFT_size = (FFT_size>>1);
	for(int f=0; f<nFFTs; f++){
		// Comparison of the first and the last elements
		float2 kFFT_tempvalue, cuFFT_tempvalue;
		kFFT_tempvalue = kFFT[f*kFFT_size];
		cuFFT_tempvalue.x = cuFFT[f*cuFFT_size].x;
		cuFFT_tempvalue.y = cuFFT[(f+1)*cuFFT_size - 1].x;
		error = get_error(kFFT_tempvalue, cuFFT_tempvalue);
		if(cislo<20 && error > max_error) {
			printf("FFT: %d; element: %d; Error is [%f] value is [%f,%f] while it should be [%f,%f]\n", f, 0, error, kFFT_tempvalue.x, kFFT_tempvalue.y, cuFFT_tempvalue.x, cuFFT_tempvalue.y);
			nErrors++;
		}
		
		for(int i=1; i<kFFT_size; i++){
			error = get_error(kFFT[f*kFFT_size + i], cuFFT[f*cuFFT_size + i]);
			total_error = total_error + error;
			if( (cislo<20 && error > max_error) && max_cislo<50){
				printf("FFT: %d; element: %d; Error is [%f] value is [%f,%f] while it should be [%f,%f]\n", f, i, error, kFFT[f*kFFT_size + i].x, kFFT[f*kFFT_size + i].y, cuFFT[f*cuFFT_size + i].x, cuFFT[f*cuFFT_size + i].y);
				cislo++;
				max_cislo++;
				nErrors++;
			}
		}
	}
	(*cumulative_error) = total_error;
	(*mean_error) = total_error/(FFT_size*nFFTs);
	return(nErrors);
}

int Compare_C2R_output(float *kFFT, float *cuFFT, int FFT_size, int nFFTs, double *cumulative_error, double *mean_error){
	float error;
	float total_error = 0;
	int nErrors = 0;
	int cislo = 0, max_cislo = 0;
	size_t pos;
	for(int f=0; f<nFFTs; f++){
		for(int i=0; i<FFT_size; i++){
			pos = f*FFT_size + i;
			kFFT[pos] = kFFT[pos]/(FFT_size>>1);
			cuFFT[pos] = cuFFT[pos]/FFT_size;
			error = get_error(kFFT[pos], cuFFT[pos]);
			total_error = total_error + error;
			if( (cislo<20 && error>max_error) && max_cislo<50){
				printf("FFT: %d; element: %d; Error is [%f] kFFT value is [%f] while it should be [%f] cuFFT/kFFT=%f;\n", f, i, error, kFFT[pos], cuFFT[pos], cuFFT[pos]/kFFT[pos] );
				cislo++;
				max_cislo++;
				nErrors++;
			}
		}
	}
	(*cumulative_error) = total_error;
	(*mean_error) = total_error/(FFT_size*nFFTs);
	return(nErrors);
}


int GPU_cuFFT_R2C(float2 *h_output, float *h_input, int FFT_size, int nFFTs, int nRuns);
int GPU_cuFFT_C2R(float *h_output, float2 *h_input, int FFT_size, int nFFTs, int nRuns);
int GPU_smFFT_R2C(float2 *h_output, float *h_input, int FFT_size, int nFFTs, int nRuns);
int GPU_smFFT_C2R(float *h_output, float2 *h_input, int FFT_size, int nFFTs, int nRuns);

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
	int nRuns    = strtol(argv[3],&pEnd,10);
	
	int FFT_C2R_size = (FFT_size>>1) + 1;
	int FFT_C2R_size_kFFT = (FFT_size>>1);
	size_t input_size_R2C        = nFFTs*FFT_size;
	size_t input_size_R2C_bytes  = input_size_R2C*sizeof(float);
	size_t input_size_C2R        = nFFTs*((FFT_size>>1)+1);
	size_t input_size_C2R_bytes  = input_size_C2R*sizeof(float2);
	
	size_t output_size_R2C       = nFFTs*((FFT_size>>1)+1);
	size_t output_size_R2C_bytes = output_size_R2C*sizeof(float2);
	size_t output_size_C2R       = nFFTs*FFT_size;
	size_t output_size_C2R_bytes = output_size_C2R*sizeof(float);
	
	if(DEBUG) printf("FFT size: %d; Number of FFTs: %d; input size = %0.3f MB; output size = %0.3f MB\n", FFT_size, nFFTs, input_size_R2C_bytes/(1024.0*1024.0), output_size_R2C_bytes/(1024.0*1024.0));
	if(FFT_size<128) { printf("This FFT works for N>=128.\n"); return(1); }
	
	double cumulative_error,mean_error;

	float  *h_input_R2C;
	float2 *h_input_C2R;
	float2 *h_input_C2R_kFFT;
	
	float2 *h_kFFT_output;
	float  *h_kFFT_output_inverse;
	float2 *h_cuFFT_output;
	float  *h_cuFFT_output_inverse;

	if (DEBUG) printf("\nHost memory allocation...\t");
		h_input_R2C 	 = (float *)malloc(input_size_R2C_bytes);
		h_input_C2R      = (float2 *)malloc(input_size_C2R_bytes);
		h_input_C2R_kFFT = (float2 *)malloc(input_size_C2R_bytes);
		
		h_kFFT_output 	       = (float2 *)malloc(output_size_R2C_bytes);
		h_kFFT_output_inverse  = (float *)malloc(output_size_C2R_bytes);
		h_cuFFT_output         = (float2 *)malloc(output_size_R2C_bytes);
		h_cuFFT_output_inverse = (float *)malloc(output_size_C2R_bytes);
	if (DEBUG) printf("done.\n");

	if (DEBUG) printf("Host memory memset...\t\t");
		memset(h_input_R2C, 0.0, input_size_R2C_bytes);
		memset(h_input_C2R, 0.0, input_size_C2R_bytes);
		memset(h_input_C2R_kFFT, 0.0, input_size_C2R_bytes);
		
		memset(h_kFFT_output, 0.0, output_size_R2C_bytes);
		memset(h_kFFT_output_inverse, 0.0, output_size_C2R_bytes);
		memset(h_cuFFT_output, 0.0, output_size_R2C_bytes);
		memset(h_cuFFT_output_inverse, 0.0, output_size_C2R_bytes);
	if (DEBUG) printf("done.\n");

	if (DEBUG) printf("Random data set...\t\t");	
		srand(time(NULL));
		for(int f = 0; f<nFFTs; f++){
			for(int s = 0; s<FFT_size; s++){
				h_input_R2C[f*FFT_size + s] = rand() / (float)RAND_MAX;
			}
		}
		
		float rnd, rnd_last;
		for(int f = 0; f<nFFTs; f++){
			rnd_last = rand() / (float)RAND_MAX;
			
			rnd = rand() / (float)RAND_MAX;
			h_input_C2R[f*FFT_C2R_size + 0].x = rnd;
			h_input_C2R[f*FFT_C2R_size + 0].y = 0;
			h_input_C2R_kFFT[f*FFT_C2R_size_kFFT + 0].x = rnd;
			h_input_C2R_kFFT[f*FFT_C2R_size_kFFT + 0].y = rnd_last;
			for(int s = 1; s<(FFT_size>>1); s++){
				rnd = rand() / (float)RAND_MAX;
				h_input_C2R[f*FFT_C2R_size + s].x = rnd;
				h_input_C2R_kFFT[f*FFT_C2R_size_kFFT + s].x = rnd;
				
				rnd = rand() / (float)RAND_MAX;
				h_input_C2R[f*FFT_C2R_size + s].y = rnd;
				h_input_C2R_kFFT[f*FFT_C2R_size_kFFT + s].y = rnd;
			}
			h_input_C2R[f*FFT_C2R_size + (FFT_size>>1)].x = rnd_last;
		}
		//Export(h_input_C2R, FFT_C2R_size, "Input_cuFFT_C2R.dat");
		//Export(h_input_C2R_kFFT, FFT_C2R_size_kFFT, "Input_kFFT_C2R.dat");
	if (DEBUG) printf("done.\n");
	
	
	//--------------------------> R2C transformation
	GPU_cuFFT_R2C(h_cuFFT_output,  h_input_R2C, FFT_size, nFFTs, nRuns);
	GPU_smFFT_R2C(  h_kFFT_output, h_input_R2C, FFT_size, nFFTs, nRuns);
	//Export(h_cuFFT_output, FFT_C2R_size, "GPU_cuFFT_R2C.dat");
	//Export(h_kFFT_output, FFT_C2R_size_kFFT, "GPU_kFFT_R2C.dat");
	#ifdef TESTING
		int nErrors = 0;
		cumulative_error = 0; mean_error = 0;
		nErrors = Compare_R2C_output(h_kFFT_output, h_cuFFT_output, FFT_size, nFFTs, &cumulative_error, &mean_error);
		if(nErrors==0) printf("  FFT test:\033[1;32mPASSED\033[0m\n");
		else printf("  FFT test:\033[1;31mFAILED\033[0m\n");
	#endif
	
	
	//--------------------------> C2R transformation
	GPU_cuFFT_C2R(h_cuFFT_output_inverse, h_input_C2R,      FFT_size, nFFTs, nRuns);
	GPU_smFFT_C2R(h_kFFT_output_inverse,  h_input_C2R_kFFT, FFT_size, nFFTs, nRuns);
	//Export(h_cuFFT_output_inverse, FFT_C2R_size, "GPU_cuFFT_C2R.dat");
	//Export(h_kFFT_output_inverse, FFT_C2R_size_kFFT, "GPU_kFFT_C2R.dat");
	#ifdef TESTING
		nErrors = 0;
		cumulative_error = 0; mean_error = 0;
		nErrors = Compare_C2R_output(h_kFFT_output_inverse, h_cuFFT_output_inverse, FFT_size, nFFTs, &cumulative_error, &mean_error);
		if(nErrors==0) printf("  FFT test:\033[1;32mPASSED\033[0m\n");
		else printf("  FFT test:\033[1;31mFAILED\033[0m\n");
	#endif
	
	
	free(h_input_R2C);
	free(h_input_C2R);
	free(h_input_C2R_kFFT);
	free(h_kFFT_output);
	free(h_kFFT_output_inverse);
	free(h_cuFFT_output);
	free(h_cuFFT_output_inverse);
	
	cudaDeviceReset();

	if (DEBUG) printf("\nFinished!\n");

	return (0);
}
