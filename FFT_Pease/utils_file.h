#include <iostream>
#include <fstream>
#include <iomanip> 

using namespace std;

long int File_size_byte(ifstream &FILEIN) {
	ifstream::pos_type size;
	FILEIN.seekg(0,ios::end);
	size=FILEIN.tellg();
	FILEIN.seekg(0,ios::beg);
	return((long int) size);
}

long int File_size_row(ifstream &FILEIN){
		std::size_t count=0;
		FILEIN.seekg(0,ios::beg);
		for(std::string line; std::getline(FILEIN, line); ++count){}
	return((long int)count);
}

bool Load_data(ifstream &FILEIN, float2 *data){
	long int count = 0;
	while (!FILEIN.eof()){
		FILEIN >> data[count].x;
		data[count].y = 0;
		count++;
	}
	return(1);
}

int Save_data(char str[], float *input, int nDMs, int nTimesamples){
	ofstream FILEOUT;
	FILEOUT.open(str);
	for(int DM=0;DM<nDMs;DM++){
		for(int Ts=0;Ts<nTimesamples;Ts++){
			FILEOUT << input[DM*nTimesamples+Ts] << " ";
		}//nDMs
		FILEOUT << endl;
	} //nTimesamples
	FILEOUT.close();
	return(1);
}

bool save_time(char str[], int nSpectra, int nSamples, float cuFFT_time, float FFT_time, float FFT_external_time, float FFT_multiple_time, float FFT_multiple_reuse_time, float FFT_multiple_reuse_registers_time, float transfer_in, float transfer_out){
	ofstream FILEOUT;
	FILEOUT.open (str, std::ofstream::out | std::ofstream::app);
	FILEOUT << std::fixed << std::setprecision(8) << nSpectra << " " << nSamples << " " << cuFFT_time << " " << FFT_time << " " << FFT_external_time << " " << FFT_multiple_time << " " << FFT_multiple_reuse_time << " " << FFT_multiple_reuse_registers_time << " " << transfer_in << " " << transfer_out <<  endl;
	FILEOUT.close();
	return 0;
}