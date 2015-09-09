/******************************************************************************
Copyright 2010-2012 Rodrigo Hernandez Cordoba

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
******************************************************************************/

#include "AeonGUI.h"
#include <cstdio>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

namespace AeonGUI
{
#ifdef USE_CUDA
    int CudaDeviceCount;
    int CudaDeviceIndex;
    cudaDeviceProp* CudaDeviceProps;
#endif
    bool Initialize()
    {
#ifdef USE_CUDA
        if ( cudaGetDeviceCount ( &CudaDeviceCount ) != cudaSuccess )
        {
            return false;
        }
        if ( cudaGetDevice ( &CudaDeviceIndex ) != cudaSuccess )
        {
            return false;
        }
        CudaDeviceProps = new cudaDeviceProp[CudaDeviceCount];
        for ( int i = 0; i < CudaDeviceCount; ++i )
        {
            if ( cudaGetDeviceProperties ( CudaDeviceProps + i, i ) != cudaSuccess )
            {
                delete [] CudaDeviceProps;
                return false;
            }
            printf ( "CUDA Device %d is%scurent device\nMax Threads Per Block: %d\nMax Thread Blocks (%d,%d,%d)\nMax Blocks per Grid (%d,%d,%d)\n", i + 1, ( i == CudaDeviceIndex ) ? " " : " not ",
                     CudaDeviceProps[i].maxThreadsPerBlock,
                     CudaDeviceProps[i].maxThreadsDim[0],
                     CudaDeviceProps[i].maxThreadsDim[1],
                     CudaDeviceProps[i].maxThreadsDim[2],
                     CudaDeviceProps[i].maxGridSize[0],
                     CudaDeviceProps[i].maxGridSize[1],
                     CudaDeviceProps[i].maxGridSize[2] );
        }
#endif
        return true;
    }
    void Finalize()
    {
#ifdef USE_CUDA
        if ( CudaDeviceProps != NULL )
        {
            delete[] CudaDeviceProps;
        }
#endif
    }
}