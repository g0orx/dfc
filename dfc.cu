/**
* @file dfc.cu
* @brief DFC code
* @author John Melton, G0ORX/N6LYT
*/


/* Copyright (C)
* 2015 - John Melton, G0ORX/N6LYT
*
* Based on code by Steven Passe AD0ES and Vasiliy Gokoyev K3IT
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
*
*/

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cufft.h>
#include <helper_cuda.h>

#include "common.cuh"
#include "filters.cuh"
#include "inputbuffer.cuh"
#include "rawbuffer.cuh"
#include "time.cuh"
#include "dfc.cuh"

static pthread_t dfcThreadId;

static cufftReal* timesamples;
static cufftReal* deviceTimesamples;
static cufftReal* delaysamples;
static cufftReal* deviceDelaysamples;

static cufftHandle planR2C;  // forward fft from real time domain samples to complex frequency domain samples

cufftComplex* frequencysamples;
cufftComplex* deviceFrequencysamples;

sem_t frequencyBufferFull;
sem_t frequencyBufferEmpty;

int captureFile;

static bool running;

void* dfcThread(void* args);

__global__ void
convertInput(const short* dataIn, cufftReal* dataOut, cufftReal* delay, short* raw) {
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadId<RAW_BUFFER_SIZE) {
        raw[threadId]=dataIn[threadId];
    }

    if(threadId<(P_SIZE-1)) {
        //overlap
        dataOut[threadId]=delay[threadId];
        delay[threadId]=dataIn[L_SIZE-(P_SIZE-1)+threadId]/DIVISOR;
    } else {
        dataOut[threadId]=dataIn[threadId-(P_SIZE-1)]/DIVISOR;
    }
}
   
__global__ void
gpu_make_analytic(cufftComplex* samples)
{
    const size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t numThreads = blockDim.x * gridDim.x;

    for (int i = threadId; i < COMPLEX_SIGNAL_SIZE; i += numThreads) {
        if (i == 0) {
            samples[i] = ComplexAdd(samples[i], samples[COMPLEX_SIGNAL_SIZE]);
        } else {
            samples[i] = ComplexScale(samples[i], 2.0f);
        }
    }
}

void initDfc(void) {
    fprintf(stderr,"initDfc: timesamples size=%ld\n",DFT_BLOCK_SIZE*sizeof(cufftReal));

    cudaError_t result = cudaHostAlloc(&timesamples, DFT_BLOCK_SIZE*sizeof(cufftReal), cudaHostAllocMapped);
    if (result != cudaSuccess) {
       fprintf(stderr, "initDfc: Error cudaHostAlloc for input samples %d\n", result);
       exit(EXIT_FAILURE);
    }

    result = cudaHostGetDevicePointer(&deviceTimesamples, timesamples, 0);
    if (result != cudaSuccess) {
       fprintf(stderr, "Error timesamples cudaHostGetDevicePointer %d\n", result);
       exit(EXIT_FAILURE);
    }

    fprintf(stderr,"initDfc: delaysamples size=%ld\n",(P_SIZE-1)*sizeof(cufftReal));
    result = cudaHostAlloc(&delaysamples, (P_SIZE-1)*sizeof(cufftReal), cudaHostAllocMapped);
    if (result != cudaSuccess) {
       fprintf(stderr, "initDfc: Error cudaHostAlloc for delay samples %d\n", result);
       exit(EXIT_FAILURE);
    }

    result = cudaHostGetDevicePointer(&deviceDelaysamples, delaysamples, 0);
    if (result != cudaSuccess) {
       fprintf(stderr, "Error delaysamples cudaHostGetDevicePointer %d\n", result);
       exit(EXIT_FAILURE);
    }


    fprintf(stderr,"initDfc: frequencysamples size=%ld\n",(COMPLEX_SIGNAL_SIZE+1)*sizeof(cufftComplex));
    result = cudaHostAlloc(&frequencysamples, (COMPLEX_SIGNAL_SIZE+1)*sizeof(cufftComplex), cudaHostAllocMapped);
    if (result != cudaSuccess) {
       fprintf(stderr, "initDfc: Error cudaHostAlloc for frequency samples %d\n", result);
       exit(EXIT_FAILURE);
    }

    result = cudaHostGetDevicePointer(&deviceFrequencysamples, frequencysamples, 0);
    if (result != cudaSuccess) {
       fprintf(stderr, "Error frequencysamples cudaHostGetDevicePointer %d\n", result);
       exit(EXIT_FAILURE);
    }

    fprintf(stderr,"initDfc: planR2C size=%d\n",DFT_BLOCK_SIZE);
    cufftResult error = cufftPlan1d(&planR2C, DFT_BLOCK_SIZE, CUFFT_R2C, 1);
    if(error!=CUFFT_SUCCESS) {
       fprintf(stderr,"Error creating cufftPlan1d for input buffer: %s\n", _cudaGetErrorEnum(error));
       exit(EXIT_FAILURE);
    }

    int res=sem_init(&frequencyBufferFull, 0, 0);
    if(res!=0) {
        fprintf(stderr,"initDfc: sem_init failed for frequencyBufferFull%d\n", result);
        exit(EXIT_FAILURE);
    }

    res=sem_init(&frequencyBufferEmpty, 0, 0);
    if(res!=0) {
        fprintf(stderr,"initDfc: sem_init failed for frequencyBufferEmpty%d\n", result);
        exit(EXIT_FAILURE);
    }

    if(capture) {
        captureFile=creat("raw.bin",O_RDWR);
        if(captureFile<0) {
            fprintf(stderr,"Cannot open capture file: result=%d \n",captureFile);
            exit(EXIT_FAILURE);
        }
    }

    res=pthread_create(&dfcThreadId, NULL, dfcThread, NULL);
    if(res<0) {
        fprintf(stderr, "Error creating DFC thread: %d\n", res);
        exit(EXIT_FAILURE);
    }
}

void dfcTerminate() {
    running=false;
    if(capture) {
        close(captureFile);
    }
}

void* dfcThread(void* args) {
    int result;
    cufftResult error;

    int rawcount=0;

#ifdef TIMING
    long long starttime;
    long long endtime;
#endif

    fprintf(stderr,"dfcThread: running on cpu %d\n", sched_getcpu());
    running=true;

    // get the first buffer
    result=sem_post(&inputBufferEmpty);
    if(result!=0) {
        fprintf(stderr, "dfcThread: sem_post failed for inputBufferEmpty: %d\n", result);
        exit(EXIT_FAILURE);
    }

    while(running) {
        result=sem_wait(&inputBufferFull);
        if(result!=0) {
            fprintf(stderr, "dfcThread: sem_wait failed for inputBufferFull: %d\n", result);
            exit(EXIT_FAILURE);
        }

        if(capture) {
            write(captureFile, inputBuffer, L_SIZE*sizeof(short));
        }

#ifdef TIMING
        starttime=current_timestamp();
#endif

        // process the buffer (convert from short to real and overlap)
        convertInput<<<DFT_BLOCK_SIZE/1024,1024>>>(deviceInputBuffer, deviceTimesamples, deviceDelaysamples, deviceRawBuffer);

/*
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if(err != cudaSuccess)
        {
          // print the CUDA error message and exit
          fprintf(stderr,"convertInput CUDA error: %s\n", cudaGetErrorString(err));
          exit(-1);
        }
*/

        // get the next buffer
        result=sem_post(&inputBufferEmpty);
        if(result!=0) {
            fprintf(stderr, "dfcThread: sem_post failed for inputBufferEmpty: %d\n", result);
            exit(EXIT_FAILURE);
        }

        // wait for frequency buffer to be empty
        result=sem_wait(&frequencyBufferEmpty);
        if(result!=0) {
            fprintf(stderr, "dfcThread: sem_wait failed for frequencyBufferEmpty: %d\n", result);
            exit(EXIT_FAILURE);
        }
        
        // forward FFT to convert from time domain to frequency domain
        error = cufftExecR2C(planR2C, deviceTimesamples, deviceFrequencysamples);
        if(error!=CUFFT_SUCCESS) {
           fprintf(stderr,"Error executing planR2C for input buffer: %s\n", _cudaGetErrorEnum(error));
           exit(EXIT_FAILURE);
        }

        gpu_make_analytic<<<COMPLEX_SIGNAL_SIZE/4096,1024>>>(deviceFrequencysamples);
/*
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if(err != cudaSuccess)
        {
          // print the CUDA error message and exit
          fprintf(stderr,"gpu_make_analytic: %s\n", cudaGetErrorString(err));
          exit(-1);
        }
*/

        // let hermes process the buffer
        result=sem_post(&frequencyBufferFull);
        if(result!=0) {
            fprintf(stderr, "dfcThread: sem_post failed for frequencyBufferFull: %d\n", result);
            exit(EXIT_FAILURE);
        }

        rawcount++;
        // pace the output of the raw data
        if(rawcount==8) {
            // wait for hermes to process last buffer
            result=sem_wait(&rawBufferEmpty);
            if(result!=0) {
                fprintf(stderr, "dfcThread: sem_wait failed for rawBufferEmpty: %d\n", result);
                exit(EXIT_FAILURE);
            }

            // let hermes process the raw buffer
            result=sem_post(&rawBufferFull);
            if(result!=0) {
                fprintf(stderr, "dfcThread: sem_post failed for rawBufferFull: %d\n", result);
                exit(EXIT_FAILURE);
            }

            rawcount=0;
        }

#ifdef TIMING
        endtime=current_timestamp();
        fprintf(stderr,"dfc took %lld ms to process a buffer\n", endtime-starttime);
#endif
        
    }
    return NULL;
}
