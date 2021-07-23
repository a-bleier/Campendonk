package main

/*
#include "kernel.cuh"
#cgo LDFLAGS: -L${SRCDIR}/cuda -lkernel
#cgo CFLAGS: -I${SRCDIR}/cuda
*/
import "C"

type config struct {
	xStart     float64
	yStart     float64
	xDim       int
	yDim       int
	xDelta     float64
	yDelta     float64
	iterations int
}

func (con *config) toC() C.Config {
	return C.Config{
		xStart:     C.double(con.xStart),
		yStart:     C.double(con.yStart),
		xDim:       C.int(con.xDim),
		yDim:       C.int(con.yDim),
		xDelta:     C.double(con.xDelta),
		yDelta:     C.double(con.yDelta),
		iterations: C.int(con.iterations),
	}
}

func test_config() {
	conf := &config{0.0, 0.0, 100, 100, 0.1, 0.1, 1000}
	C.print_config(conf.toC())

}

func test_mandelbrot() {

	conf := &config{0.0, 0.0, 100, 100, 0.01, 0.01, 1000}
	out := make([]C.int, conf.xDim*conf.yDim)
	pointer := (&out[0])
	C.launch_mandelbrot(conf.toC(), pointer)

}
