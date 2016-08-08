#!/bin/bash

make;
./FFT.exe 64 100000
./FFT.exe 128 100000
./FFT.exe 256 100000
./FFT.exe 512 100000
./FFT.exe 1024 100000