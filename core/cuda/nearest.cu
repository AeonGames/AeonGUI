#include <stdio.h>
#include "cuda_functions.h"

namespace AeonGUI
{
__global__ void nearest()
{
    printf("nearest %u\n",threadIdx.x);
}

void NearestNeighbour()
{
    nearest<<<16,1>>>();
    cudaDeviceSynchronize();
}
}
