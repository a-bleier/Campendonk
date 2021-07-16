#!/bin/bash
nvcc --ptxas-options=-v --compiler-options '-fPIC' -o cuda/libkernel.so --shared cuda/kernel.cu
go build -o bin/campendonk