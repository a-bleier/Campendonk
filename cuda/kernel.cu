//TODO: give a thread multiple points
//TODO: let the thread calculate a color

extern "C"
{
#include <stdio.h>
#include <cuda.h>
#include <stdio.h>
#include "kernel.cuh"
#include "common.cuh"
    __global__ void test()
    {
        printf("Hello from cuda Core !\n");
    }

    __global__ void mandelbrote_kernel(int *res, int len, double xDelta, double yDelta, double xStart, double yStart, int iterations)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        double x_c = col * xDelta + xStart;
        double y_c = row * yDelta + yStart;
        double x = 0;
        double y = 0;
        double x_old = 0;
        double y_old = 0;
        //iterate this point, check if it's in the set or not; write a result in res
        for(int i=0; i<iterations; i++){
            x = x_old * x_old - y_old * y_old + x_c;
            y = 2 * x_old * y_old + y_c;
            double d_sq = x * x + y * y;

            if(d_sq >= 9){
                res[col + gridDim.x * blockDim.x * row] = i+1;
                return;
            }
            x_old = x;
            y_old = y;
        }
        res[col + gridDim.x * blockDim.x * row] = -1;
    }

    void launch_test()
    {

        // Call the kernel:
        test<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

    void launch_mandelbrot(Config con, int *out)
    {
        int *res;
        int len = con.xDim * con.yDim;
        CHECK(cudaMalloc((int **)&res, len * sizeof(int)));
        cudaMemset(res, 0, len);

        //build cluster
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(ceilf((float) con.xDim / (float) BLOCK_SIZE), ceilf((float) con.yDim / (float) BLOCK_SIZE));

        CHECK(cudaDeviceSynchronize());

        mandelbrote_kernel<<<grid, block>>>(res, len, con.xDelta, con.yDelta, con.xStart, con.yStart, con.iterations);

        CHECK(cudaDeviceSynchronize());

        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(out, res, len * sizeof(int), cudaMemcpyDeviceToHost));
        
        // for(int i=1; i<len+1; i++){
        //     printf("%d\t", out[i-1]);
        //     if(i % 10 == 0) printf("\n");
        // }

        printf("here\n");
        CHECK(cudaFree(res));
    }

    void print_config(Config c)
    {
    }
}
