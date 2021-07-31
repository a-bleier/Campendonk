package main

/*
#include "kernel.cuh"
#cgo LDFLAGS: -L${SRCDIR}/cuda -lkernel
#cgo CFLAGS: -I${SRCDIR}/cuda
*/
import "C"
import (
	"image"
	"image/color"
	"image/png"
	"os"
)

type config struct {
	xStart      float64
	yStart      float64
	xDim        int
	yDim        int
	xDelta      float64
	yDelta      float64
	xResolution int
	yResolution int
	iterations  int
}

func (con *config) toC() C.Config {
	return C.Config{
		xStart:      C.double(con.xStart),
		yStart:      C.double(con.yStart),
		xDim:        C.int(con.xDim),
		yDim:        C.int(con.yDim),
		xDelta:      C.double(con.xDelta),
		yDelta:      C.double(con.yDelta),
		xResolution: C.int(con.xResolution),
		yResolution: C.int(con.yResolution),
		iterations:  C.int(con.iterations),
	}
}

func test_config() {
	conf := &config{100, 0.0, 100, 100, 0.1, -0.1, 1920, 1080, 1000}
	C.print_config(conf.toC())

}

func test_mandelbrot() {
	width, height := 512, 512
	xRes, yRes := 1024, 1024

	conf := &config{-1, 1, width, height, 0.01, -0.01, 1024, 1024, 1000} //testing: every point gets its pixel
	out := make([]C.char, conf.xResolution*conf.yResolution*3)
	pointer := (&out[0])
	C.launch_mandelbrot(conf.toC(), pointer)

	upleft := image.Point{0, 0}
	downright := image.Point{xRes, yRes}

	img := image.NewRGBA(image.Rectangle{upleft, downright})

	for y := 0; y < conf.yResolution; y++ {
		for x := 0; x < conf.xResolution; x++ {
			index := y*conf.xResolution + x
			img.Set(x, y, color.RGBA{
				uint8(out[index*3]),
				uint8(out[index*3+1]),
				uint8(out[index*3+2]),
				0xff,
			})

		}
	}

	f, _ := os.Create("out.png")
	png.Encode(f, img)
}
