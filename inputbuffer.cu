#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <cufft.h>
#include <helper_cuda.h>

#include "common.cuh"
#include "inputbuffer.cuh"


short* inputBuffer;
short* deviceInputBuffer;

sem_t inputBufferEmpty;
sem_t inputBufferFull;

void initInputBuffer() {
    fprintf(stderr,"initInputBuffer: size=%ld\n",L_SIZE*sizeof(short));
    cudaError_t result = cudaHostAlloc(&inputBuffer, L_SIZE*sizeof(short), cudaHostAllocMapped);
    if (result != cudaSuccess) {
       fprintf(stderr, "Error inputBuffer cudaHostAlloc %d\n", result);
       exit(EXIT_FAILURE);
    }

    result = cudaHostGetDevicePointer(&deviceInputBuffer, inputBuffer, 0);
    if (result != cudaSuccess) {
       fprintf(stderr, "Error inputBuffer cudaHostGetDevicePointer %d\n", result);
       exit(EXIT_FAILURE);
    }


    int res=sem_init(&inputBufferEmpty, 0, 0);
    if(result!=0) {
        fprintf(stderr,"inputbuffer: sem_init failed for inputBufferEmpty%d\n", result);
        exit(EXIT_FAILURE);
    }

    res=sem_init(&inputBufferFull, 0, 0);
    if(result!=0) {
        fprintf(stderr,"inputbuffer: sem_init failed for inputBufferFull%d\n", result);
        exit(EXIT_FAILURE);
    }

}
