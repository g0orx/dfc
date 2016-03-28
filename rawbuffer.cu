#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <cufft.h>
#include <helper_cuda.h>

#include "rawbuffer.cuh"

short* rawBuffer;
short* deviceRawBuffer;

sem_t rawBufferEmpty;
sem_t rawBufferFull;

void initRawBuffer() {
    fprintf(stderr,"initRawBuffer\n");
    cudaError_t result = cudaHostAlloc(&rawBuffer, RAW_BUFFER_SIZE*sizeof(short), cudaHostAllocMapped);
    if (result != cudaSuccess) {
       fprintf(stderr, "Error rawBuffer cudaHostAlloc %d\n", result);
       exit(EXIT_FAILURE);
    }

    result = cudaHostGetDevicePointer(&deviceRawBuffer, rawBuffer, 0);
    if (result != cudaSuccess) {
       fprintf(stderr, "Error rawBuffer cudaHostGetDevicePointer %d\n", result);
       exit(EXIT_FAILURE);
    }


    int res=sem_init(&rawBufferEmpty, 0, 0);
    if(result!=0) {
        fprintf(stderr,"rawbuffer: sem_init failed for rawBufferEmpty%d\n", result);
        exit(EXIT_FAILURE);
    }

    res=sem_init(&rawBufferFull, 0, 0);
    if(result!=0) {
        fprintf(stderr,"rawbuffer: sem_init failed for rawBufferFull%d\n", result);
        exit(EXIT_FAILURE);
    }

}
