package main

/*
#include "kernel.cuh"
#cgo LDFLAGS: -L${SRCDIR}/cuda -lkernel
#cgo CFLAGS: -I${SRCDIR}/cuda
*/
import "C"
import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
)

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
	conf := &config{100, 0.0, 100, 100, 0.1, -0.1, 1000}
	C.print_config(conf.toC())

}

func test_mandelbrot() {
	width, height := 256, 256

	conf := &config{-1, 1, width, height, 0.01, -0.01, 1000}
	out := make([]C.int, conf.xDim*conf.yDim)
	pointer := (&out[0])
	C.launch_mandelbrot(conf.toC(), pointer)
	fmt.Println(out)

	upleft := image.Point{0, 0}
	downright := image.Point{width, height}

	img := image.NewRGBA(image.Rectangle{upleft, downright})

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			iterations := int(out[y*width+x])
			if iterations == -1 {
				img.Set(x, y, color.RGBA{0, 0, 0, 0xff})
			} else {
				img.Set(x, y, color.RGBA{
					uint8(2*iterations + 30),
					uint8(3*iterations + 20),
					uint8(4*iterations + 10),
					// 0xff, 0xff, 0xff,
					0xff,
				})
			}
		}
	}

	f, _ := os.Create("out.png")
	png.Encode(f, img)
}
