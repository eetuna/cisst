/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: svlVideoCodecUDP.cu 1236 2010-02-26 20:38:21Z adeguet1 $
  
  Author(s):  Min Yang Jung
  Created on: 2010-05-24

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#include <stdio.h>
#include <cutil_inline.h>

#define CUDA_DEBUG

unsigned int BufferSize = 0;
unsigned int SignBufferSize = 0;

unsigned char * imageCurrent_d = 0;
unsigned char * imagePrev_d = 0;
unsigned char * imageDiff_d = 0;
unsigned char * imageDiffSign_d = 0;

// CUDA configuration
const int ThreadCountPerBlock = 1 << 9; // 512 threads per block
const int BlockCount = 1 << 5; // Warp size: 32

// Timer
unsigned int timer = 0;

extern "C" void InitializeCUDA(void)
{
    //
}

// Allocate image buffers on device memory
extern "C" void CreateDeviceBuffers(unsigned int bufferSize)
{
    BufferSize = bufferSize;
    SignBufferSize = bufferSize / 8 + 3;

    cutilSafeCall(cudaMalloc((void**)&imageCurrent_d, BufferSize));
    cutilSafeCall(cudaMalloc((void**)&imagePrev_d, BufferSize));
    cutilSafeCall(cudaMalloc((void**)&imageDiff_d, BufferSize));
    cutilSafeCall(cudaMalloc((void**)&imageDiffSign_d, SignBufferSize));

    cutilSafeCall(cudaMemset(imageCurrent_d, 0, BufferSize));
    cutilSafeCall(cudaMemset(imagePrev_d, 0, BufferSize));
    cutilSafeCall(cudaMemset(imageDiff_d, 0, BufferSize));
    cutilSafeCall(cudaMemset(imageDiffSign_d, 0, SignBufferSize));

#ifdef CUDA_DEBUG
    printf("CUDA - Device memory allocated: %u, %u\n", BufferSize, SignBufferSize);
#endif
}

extern "C" void CleanupDeviceBuffers()
{
    // Free device memory
    if (imageCurrent_d) cudaFree(imageCurrent_d);
    if (imagePrev_d) cudaFree(imagePrev_d);
    if (imageDiff_d) cudaFree(imageDiff_d);
    if (imageDiffSign_d) cudaFree(imageDiffSign_d);

    cutilSafeCall(cudaThreadExit());
    
#ifdef CUDA_DEBUG
    printf("CUDA - Device memory deallocated\n");
#endif    
}

// Device code that executes on the CUDA device
// Get interframe image difference in YUV space
__global__ void FrameDiff(
    unsigned char * imgCurr, unsigned char * imgPrev,
    unsigned char * imgDiff, unsigned char * imgDiffSign, int N)
{
#define STRIDE     1
#define OFFSET     0

    int n_elem_per_thread = N / (gridDim.x * blockDim.x);
    int block_start_idx = n_elem_per_thread * blockIdx.x * blockDim.x;
    int thread_start_idx = block_start_idx
        + (threadIdx.x / STRIDE) * n_elem_per_thread * STRIDE
        + ((threadIdx.x + OFFSET) % STRIDE);
    int thread_end_idx = thread_start_idx + n_elem_per_thread * STRIDE;
    if (thread_end_idx > N) thread_end_idx = N;

    int diff, signIndex, bitShift;
    for(int idx = thread_start_idx; idx < thread_end_idx; idx += STRIDE)
    {
        diff = imgCurr[idx] - imgPrev[idx];

        signIndex = idx / 8;
        bitShift = idx % 8;

        imgDiffSign[signIndex] |= (diff > 0 ? 0 : 1) << bitShift;

        imgDiff[idx] = abs(diff);
        imgPrev[idx] += diff;
    }
}

extern "C" void InterFrameDiff(unsigned char * imageCurrent,
                               unsigned char * imagePrevious,
                               unsigned char * imageDiff,
                               unsigned char * imageDiffSign)
{
    // Copy host memory content to device memory
    cutilSafeCall(cudaMemcpy(imagePrev_d,     imagePrevious, BufferSize,     cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(imageCurrent_d,  imageCurrent,  BufferSize,     cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(imageDiffSign_d, imageDiffSign, SignBufferSize, cudaMemcpyHostToDevice));
    
    // Timer: tic
    cutCreateTimer(&timer);
    cutStartTimer(timer);

    // Do calculation on device
    FrameDiff<<<BlockCount, ThreadCountPerBlock>>>(
        imageCurrent_d, imagePrev_d, imageDiff_d, imageDiffSign_d, BufferSize);

    // Wait for calculation to finish on CUDA
    cudaThreadSynchronize();

    // Timer: toc
    cutStopTimer(timer);
    
    printf("CUDA - execution time: %f msec\n", cutGetTimerValue(timer));

    // Retrieve result from device memory to host memory
    cutilSafeCall(cudaMemcpy(imageDiff,     imageDiff_d,     BufferSize,     cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(imageDiffSign, imageDiffSign_d, SignBufferSize, cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(imagePrevious, imagePrev_d,     BufferSize,     cudaMemcpyDeviceToHost));
}

