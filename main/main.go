package main

/*
#include "kernel.cuh"
#cgo LDFLAGS: -L${SRCDIR}/../cuda -lkernel
#cgo CFLAGS: -I${SRCDIR}/../cuda
*/
import "C"

func test() {
	C.launch_test()
}

func main() {
	test()
}
