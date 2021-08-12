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
	"math"
	"os"

	"gioui.org/f32"
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

func (conf *config) calculate(dim image.Point, cim f32.Point, zoom float64) image.Image {
	//TODO determine texture granularity
	//TODO determine zoom factor
	//TODO determine center
	//TODO I need old config to determine new center

	/*
		window -> old set frame -> +shift
	*/

	conf.xDim, conf.yDim = dim.X, dim.Y

	//click in set
	cis_x := conf.xStart + float64(conf.xDim)*float64(cim.X)/float64(dim.X)*math.Abs(conf.xDelta)
	cis_y := conf.yStart - float64(conf.yDim)*float64(cim.Y)/float64(dim.Y)*math.Abs(conf.yDelta)

	fmt.Println("Click in set: ")
	fmt.Println(cis_x, " ", cis_y)

	// x_shift := cis_x - (conf.xStart + 0.5*float64(conf.xDim)*conf.xDelta)
	// y_shift := cis_y - (conf.yStart + 0.5*float64(conf.yDim)*conf.yDelta)

	conf.xDelta *= zoom
	conf.yDelta *= zoom

	conf.xStart = cis_x - 0.5*float64(conf.xDim)*math.Abs(conf.xDelta)
	conf.yStart = cis_y + 0.5*float64(conf.yDim)*math.Abs(conf.yDelta)

	conf.xResolution, conf.yResolution = dim.X, dim.Y

	out := make([]C.char, conf.xResolution*conf.yResolution*3)
	pointer := (&out[0])
	C.launch_mandelbrot(conf.toC(), pointer)

	img := image.NewRGBA(image.Rectangle{image.Point{0, 0}, dim})

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

	return img
}

func test_mandelbrot() image.Image {
	width, height := 2048, 2048
	xRes, yRes := 1024, 1024

	conf := &config{-2, 2, width, height, 0.001, -0.001, 1024, 1024, 1000} //testing: every point gets its pixel
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

	return img
}
