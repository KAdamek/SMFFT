#include <fftw3.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void Normalize_FFT(float2 *output, int nChannels, int nSpectra, double factor){
	int s,c;
	for(s=0;s<nSpectra;s++){
		for(c=0;c<nSpectra;c++){
			output[s*nChannels+c].x=output[s*nChannels+c].x/factor;
			output[s*nChannels+c].y=output[s*nChannels+c].y/factor;
		}
	}
}


void FIR_FFT_check(float2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error){
	float2 *spectra;
	spectra	= (float2 *)malloc(nChannels*sizeof(float2));
	double etemp=0;
	double error=0;
	for(int bl=0;bl<nSpectra;bl++){
		for(int c=0;c<nChannels;c++){
			spectra[c].x=0.0;spectra[c].y=0.0;
			for(int t=0;t<nTaps;t++){
				spectra[c].x+=coeff[t*nChannels + c]*input_data[bl*nChannels + t*nChannels + c].x;
				spectra[c].y+=coeff[t*nChannels + c]*input_data[bl*nChannels + t*nChannels + c].y;
			}//nTaps
		}//nChannels
		Perform_FFT(spectra, nChannels);
		
		for(int c=0;c<nChannels;c++){	
			etemp=abs(spectra[c].x-spectra_GPU[bl*nChannels + c].x);
			error+=etemp;
			etemp=abs(spectra[c].y-spectra_GPU[bl*nChannels + c].y);
			error+=etemp;
		}
		
	} //nSpectra
	*cumulative_error=error;
	*mean_error=error/(nChannels*nSpectra);
	delete[] spectra;
}
