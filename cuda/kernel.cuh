#pragma once
#include <stdio.h>
#define BLOCK_SIZE 32

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

typedef struct {
	unsigned char r;
	unsigned char g;
	unsigned char b;
} Color;

void print_config(Config c);

void launch_mandelbrot(Config con, char *out);
