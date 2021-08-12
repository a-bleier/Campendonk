package main

import (
	"fmt"
	"image"
	"log"
	"os"

	"gioui.org/app"
	"gioui.org/f32"
	"gioui.org/io/pointer"
	"gioui.org/io/system"
	"gioui.org/layout"
	"gioui.org/op"
	"gioui.org/op/paint"
	"gioui.org/widget"
)

/*
possible events:
user resizes window
user zooms in or out

resizing:
Can I scale and crop the image when resizing ?
--> widget.Fill
When to generate a new image with the device ?
--> when the user makes a right click

zoom in / out
Calculate new image on device
*/

func main() {
	img := test_mandelbrot()
	location := f32.Pt(0, 0)
	// targetLocation := location
	//configuration of the data to be calculated
	conf := &config{0.0, 0.0, 1000, 1000, 0.001, -0.001, 1000, 1000, 1000}
	// conf := &config{-2, 2, 1000, 1000, 0.001, -0.001, 1000, 1000, 1000}
	// conf.iterations = 1000
	// conf.xDelta, conf.yDelta = 1, 1
	// conf.kk
	mainLoop(func(gtx layout.Context) {
		// ops *op.Ops, queue event.Queue, windowSize image.Point
		ops := gtx.Ops

		// register area for input events
		pointer.Rect(image.Rectangle{Max: gtx.Constraints.Max}).Add(ops)

		// register the area for pointer events
		pointer.InputOp{
			Tag:   &location,
			Types: pointer.Press,
		}.Add(ops)

		//behaviour of the image
		queue := gtx.Queue
		imageOp := paint.NewImageOp(img)
		widget.Image{
			Src:      imageOp,
			Fit:      widget.Fill,
			Position: layout.NW,
		}.Layout(gtx)

		for _, ev := range queue.Events(&location) {
			switch ev := ev.(type) {
			case pointer.Event:
				if ev.Type == pointer.Press {
					targetLocation := ev.Position
					// fmt.Println(targetLocation)
					if ev.Buttons.Contain(pointer.ButtonPrimary) { //presumably left click
						fmt.Println("left click")
						//TODO zoom in
						img = conf.calculate(gtx.Constraints.Max, targetLocation, 0.5)
					} else if ev.Buttons.Contain(pointer.ButtonSecondary) { //presumably right click
						fmt.Println("right click")
						//TODO zoom out
						img = conf.calculate(gtx.Constraints.Max, targetLocation, 2.0)
					} else if ev.Buttons.Contain(pointer.ButtonTertiary) { //presumably middle click
						fmt.Println("middle click")
						//TODO adjust aspect ratio
						img = conf.calculate(gtx.Constraints.Max, f32.Point{float32(gtx.Constraints.Max.X) / 2, float32(gtx.Constraints.Max.Y) / 2}, 1.0)
					}
					fmt.Println(conf)
					imageOp = paint.NewImageOp(img)
					widget.Image{
						Src:      imageOp,
						Fit:      widget.Fill,
						Position: layout.NW,
					}.Layout(gtx)

				}
			}
		}
	})

}

func mainLoop(fn func(gtx layout.Context)) {
	go func() {
		w := app.NewWindow()
		// ops will be used to encode different operations.
		var ops op.Ops

		// listen for events happening on the window.
		for e := range w.Events() {
			// detect the type of the event.
			switch e := e.(type) {
			// this is sent when the application should re-render.
			case system.FrameEvent:
				// gtx is used to pass around rendering and event information.
				gtx := layout.NewContext(&ops, e)
				// render content
				fn(gtx)
				// render and handle the operations from the UI.
				e.Frame(gtx.Ops)

			// this is sent when the application is closed.
			case system.DestroyEvent:
				if e.Err != nil {
					log.Println(e.Err)
					os.Exit(1)
				}
				os.Exit(0)
			}
		}
	}()
	app.Main()
}
