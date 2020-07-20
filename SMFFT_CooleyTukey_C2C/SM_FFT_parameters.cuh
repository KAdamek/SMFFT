class FFT_Params {
public:
	static const int fft_exp = -1;
	static const int fft_length = -1;
	static const int warp = 32;
};

class FFT_32_forward : public FFT_Params {
	public:
	static const int fft_exp = 5;
	static const int fft_sm_required = 128;
	static const int fft_length = 128;
	static const int fft_length_quarter = 32;
	static const int fft_length_half = 64;
	static const int fft_length_three_quarters = 96;
	static const int fft_direction = 0;
	static const int fft_reorder = 1;
};

class FFT_32_forward_noreorder : public FFT_Params {
	public:
	static const int fft_exp = 5;
	static const int fft_sm_required = 128;
	static const int fft_length = 128;
	static const int fft_length_quarter = 32;
	static const int fft_length_half = 64;
	static const int fft_length_three_quarters = 96;
	static const int fft_direction = 0;
	static const int fft_reorder = 0;
};

class FFT_32_inverse : public FFT_Params {
	public:
	static const int fft_exp = 5;
	static const int fft_sm_required = 128;
	static const int fft_length = 128;
	static const int fft_length_quarter = 32;
	static const int fft_length_half = 64;
	static const int fft_length_three_quarters = 96;
	static const int fft_direction = 1;
	static const int fft_reorder = 1;
};

class FFT_32_inverse_noreorder : public FFT_Params {
	public:
	static const int fft_exp = 5;
	static const int fft_sm_required = 128;
	static const int fft_length = 128;
	static const int fft_length_quarter = 32;
	static const int fft_length_half = 64;
	static const int fft_length_three_quarters = 96;
	static const int fft_direction = 1;
	static const int fft_reorder = 0;
};

class FFT_64_forward : public FFT_Params {
	public:
	static const int fft_exp = 6;
	static const int fft_sm_required = 132;
	static const int fft_length = 128;
	static const int fft_length_quarter = 32;
	static const int fft_length_half = 64;
	static const int fft_length_three_quarters = 96;
	static const int fft_direction = 0;
	static const int fft_reorder = 1;
};

class FFT_64_forward_noreorder : public FFT_Params {
	public:
	static const int fft_exp = 6;
	static const int fft_sm_required = 132;
	static const int fft_length = 128;
	static const int fft_length_quarter = 32;
	static const int fft_length_half = 64;
	static const int fft_length_three_quarters = 96;
	static const int fft_direction = 0;
	static const int fft_reorder = 0;
};

class FFT_64_inverse : public FFT_Params {
	public:
	static const int fft_exp = 6;
	static const int fft_sm_required = 132;
	static const int fft_length = 128;
	static const int fft_length_quarter = 32;
	static const int fft_length_half = 64;
	static const int fft_length_three_quarters = 96;
	static const int fft_direction = 1;
	static const int fft_reorder = 1;
};

class FFT_64_inverse_noreorder : public FFT_Params {
	public:
	static const int fft_exp = 6;
	static const int fft_sm_required = 132;
	static const int fft_length = 128;
	static const int fft_length_quarter = 32;
	static const int fft_length_half = 64;
	static const int fft_length_three_quarters = 96;
	static const int fft_direction = 1;
	static const int fft_reorder = 0;
};

class FFT_128_forward : public FFT_Params {
	public:
	static const int fft_exp = 7;
	static const int fft_sm_required = 132;
	static const int fft_length = 128;
	static const int fft_length_quarter = 32;
	static const int fft_length_half = 64;
	static const int fft_length_three_quarters = 96;
	static const int fft_direction = 0;
	static const int fft_reorder = 1;
};

class FFT_128_forward_noreorder : public FFT_Params {
	public:
	static const int fft_exp = 7;
	static const int fft_sm_required = 132;
	static const int fft_length = 128;
	static const int fft_length_quarter = 32;
	static const int fft_length_half = 64;
	static const int fft_length_three_quarters = 96;
	static const int fft_direction = 0;
	static const int fft_reorder = 0;
};

class FFT_128_inverse : public FFT_Params {
	public:
	static const int fft_exp = 7;
	static const int fft_sm_required = 132;
	static const int fft_length = 128;
	static const int fft_length_quarter = 32;
	static const int fft_length_half = 64;
	static const int fft_length_three_quarters = 96;
	static const int fft_direction = 1;
	static const int fft_reorder = 1;
};

class FFT_128_inverse_noreorder : public FFT_Params {
	public:
	static const int fft_exp = 7;
	static const int fft_sm_required = 132;
	static const int fft_length = 128;
	static const int fft_length_quarter = 32;
	static const int fft_length_half = 64;
	static const int fft_length_three_quarters = 96;
	static const int fft_direction = 1;
	static const int fft_reorder = 0;
};

class FFT_256_forward : public FFT_Params {
	public:
	static const int fft_exp = 8;
	static const int fft_sm_required = 264;
	static const int fft_length = 256;
	static const int fft_length_quarter = 64;
	static const int fft_length_half = 128;
	static const int fft_length_three_quarters = 192;
	static const int fft_direction = 0;
	static const int fft_reorder = 1;
};

class FFT_256_forward_noreorder : public FFT_Params {
	public:
	static const int fft_exp = 8;
	static const int fft_sm_required = 264;
	static const int fft_length = 256;
	static const int fft_length_quarter = 64;
	static const int fft_length_half = 128;
	static const int fft_length_three_quarters = 192;
	static const int fft_direction = 0;
	static const int fft_reorder = 0;
};

class FFT_256_inverse : public FFT_Params {
	public:
	static const int fft_exp = 8;
	static const int fft_sm_required = 264;
	static const int fft_length = 256;
	static const int fft_length_quarter = 64;
	static const int fft_length_half = 128;
	static const int fft_length_three_quarters = 192;
	static const int fft_direction = 1;
	static const int fft_reorder = 1;
};

class FFT_256_inverse_noreorder : public FFT_Params {
	public:
	static const int fft_exp = 8;
	static const int fft_sm_required = 264;
	static const int fft_length = 256;
	static const int fft_length_quarter = 64;
	static const int fft_length_half = 128;
	static const int fft_length_three_quarters = 192;
	static const int fft_direction = 1;
	static const int fft_reorder = 0;
};

class FFT_512_forward : public FFT_Params {
	public:
	static const int fft_exp = 9;
	static const int fft_sm_required = 528;
	static const int fft_length = 512;
	static const int fft_length_quarter = 128;
	static const int fft_length_half = 256;
	static const int fft_length_three_quarters = 384;
	static const int fft_direction = 0;
	static const int fft_reorder = 1;
};

class FFT_512_forward_noreorder : public FFT_Params {
	public:
	static const int fft_exp = 9;
	static const int fft_sm_required = 528;
	static const int fft_length = 512;
	static const int fft_length_quarter = 128;
	static const int fft_length_half = 256;
	static const int fft_length_three_quarters = 384;
	static const int fft_direction = 0;
	static const int fft_reorder = 0;
};

class FFT_512_inverse : public FFT_Params {
	public:
	static const int fft_exp = 9;
	static const int fft_sm_required = 528;
	static const int fft_length = 512;
	static const int fft_length_quarter = 128;
	static const int fft_length_half = 256;
	static const int fft_length_three_quarters = 384;
	static const int fft_direction = 1;
	static const int fft_reorder = 1;
};

class FFT_512_inverse_noreorder : public FFT_Params {
	public:
	static const int fft_exp = 9;
	static const int fft_sm_required = 528;
	static const int fft_length = 512;
	static const int fft_length_quarter = 128;
	static const int fft_length_half = 256;
	static const int fft_length_three_quarters = 384;
	static const int fft_direction = 1;
	static const int fft_reorder = 0;
};

class FFT_1024_forward : public FFT_Params {
	public:
	static const int fft_exp = 10;
	static const int fft_sm_required = 1056;
	static const int fft_length = 1024;
	static const int fft_length_quarter = 256;
	static const int fft_length_half = 512;
	static const int fft_length_three_quarters = 768;
	static const int fft_direction = 0;
	static const int fft_reorder = 1;
};

class FFT_1024_forward_noreorder : public FFT_Params {
	public:
	static const int fft_exp = 10;
	static const int fft_sm_required = 1056;
	static const int fft_length = 1024;
	static const int fft_length_quarter = 256;
	static const int fft_length_half = 512;
	static const int fft_length_three_quarters = 768;
	static const int fft_direction = 0;
	static const int fft_reorder = 0;
};

class FFT_1024_inverse : public FFT_Params {
	public:
	static const int fft_exp = 10;
	static const int fft_sm_required = 1056;
	static const int fft_length = 1024;
	static const int fft_length_quarter = 256;
	static const int fft_length_half = 512;
	static const int fft_length_three_quarters = 768;
	static const int fft_direction = 1;
	static const int fft_reorder = 1;
};

class FFT_1024_inverse_noreorder : public FFT_Params {
	public:
	static const int fft_exp = 10;
	static const int fft_sm_required = 1056;
	static const int fft_length = 1024;
	static const int fft_length_quarter = 256;
	static const int fft_length_half = 512;
	static const int fft_length_three_quarters = 768;
	static const int fft_direction = 1;
	static const int fft_reorder = 0;
};

class FFT_2048_forward : public FFT_Params {
	public:
	static const int fft_exp = 11;
	static const int fft_sm_required = 2112;
	static const int fft_length = 2048;
	static const int fft_length_quarter = 512;
	static const int fft_length_half = 1024;
	static const int fft_length_three_quarters = 1536;
	static const int fft_direction = 0;
	static const int fft_reorder = 1;
};

class FFT_2048_forward_noreorder : public FFT_Params {
	public:
	static const int fft_exp = 11;
	static const int fft_sm_required = 2112;
	static const int fft_length = 2048;
	static const int fft_length_quarter = 512;
	static const int fft_length_half = 1024;
	static const int fft_length_three_quarters = 1536;
	static const int fft_direction = 0;
	static const int fft_reorder = 0;
};

class FFT_2048_inverse : public FFT_Params {
	public:
	static const int fft_exp = 11;
	static const int fft_sm_required = 2112;
	static const int fft_length = 2048;
	static const int fft_length_quarter = 512;
	static const int fft_length_half = 1024;
	static const int fft_length_three_quarters = 1536;
	static const int fft_direction = 1;
	static const int fft_reorder = 1;
};

class FFT_2048_inverse_noreorder : public FFT_Params {
	public:
	static const int fft_exp = 11;
	static const int fft_sm_required = 2112;
	static const int fft_length = 2048;
	static const int fft_length_quarter = 512;
	static const int fft_length_half = 1024;
	static const int fft_length_three_quarters = 1536;
	static const int fft_direction = 1;
	static const int fft_reorder = 0;
};

class FFT_4096_forward : public FFT_Params {
	public:
	static const int fft_exp = 12;
	static const int fft_sm_required = 4224;
	static const int fft_length = 4096;
	static const int fft_length_quarter = 1024;
	static const int fft_length_half = 2048;
	static const int fft_length_three_quarters = 3072;
	static const int fft_direction = 0;
	static const int fft_reorder = 1;
};

class FFT_4096_forward_noreorder : public FFT_Params {
	public:
	static const int fft_exp = 12;
	static const int fft_sm_required = 4224;
	static const int fft_length = 4096;
	static const int fft_length_quarter = 1024;
	static const int fft_length_half = 2048;
	static const int fft_length_three_quarters = 3072;
	static const int fft_direction = 0;
	static const int fft_reorder = 0;
};

class FFT_4096_inverse : public FFT_Params {
	public:
	static const int fft_exp = 12;
	static const int fft_sm_required = 4224;
	static const int fft_length = 4096;
	static const int fft_length_quarter = 1024;
	static const int fft_length_half = 2048;
	static const int fft_length_three_quarters = 3072;
	static const int fft_direction = 1;
	static const int fft_reorder = 1;
};

class FFT_4096_inverse_noreorder : public FFT_Params {
	public:
	static const int fft_exp = 12;
	static const int fft_sm_required = 4224;
	static const int fft_length = 4096;
	static const int fft_length_quarter = 1024;
	static const int fft_length_half = 2048;
	static const int fft_length_three_quarters = 3072;
	static const int fft_direction = 0;
	static const int fft_reorder = 0;
};
