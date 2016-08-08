#!/bin/bash

make;
./FFT.exe 128 100000
./FFT.exe 256 100000
./FFT.exe 512 100000
./FFT.exe 1024 100000
./FFT.exe 2048 50000
./FFT.exe 4096 50000
