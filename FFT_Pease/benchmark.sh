#!/bin/bash

echo "#define RADIX 3" > params.h
make
./FFT.exe 96 100000
./FFT.exe 192 100000
./FFT.exe 384 100000
./FFT.exe 768 100000
./FFT.exe 1536 100000

echo "#define RADIX 5" > params.h
make
./FFT.exe 160 100000
./FFT.exe 320 100000
./FFT.exe 640 100000
./FFT.exe 1280 100000

echo "#define RADIX 7" > params.h
make
./FFT.exe 224 100000
./FFT.exe 448 100000
./FFT.exe 896 100000
./FFT.exe 1792 100000
