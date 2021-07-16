
#include <stdio.h>
#include <cuda.h>
#include <stdio.h>

__global__ void test()
{
    printf( "Hello from cuda Core !\n");
}

extern "C" {

    void launch_test() {

        // Call the kernel:
        test<<<1,1>>>();
        cudaDeviceSynchronize();

    }

}