#pragma once
#include <stdio.h>
#define BLOCK_SIZE 32
void launch_test();

typedef struct {
	double xStart;
	double yStart;
	int xDim;
	int yDim;
	double xDelta;
	double yDelta;
	int xResolution;
	int yResolution;
	int iterations;
} Config;

void print_config(Config c);

void launch_mandelbrot(Config con, char *out);
