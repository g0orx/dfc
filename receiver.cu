/**
* @file receiver.cu
* @brief Implement a receiver
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
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/if_ether.h>
#include <netpacket/packet.h>
#include <net/if_packet.h>
#include <cufft.h>
#include <helper_cuda.h>

#include "common.cuh"
#include "receiver.cuh"
#include "dfc.cuh"
#include "inputbuffer.cuh"
#include "rawbuffer.cuh"
#include "filters.cuh"
#include "hermes.cuh"
#include "time.cuh"

#define SCALE_FACTOR  8388607.0

float scale_factor=0.25F;

RECEIVER receiver[MAX_RECEIVER];
void* receiverThread(void* arg);

void initReceiver(int rx) {
    int result;
    cudaError_t error;
    RECEIVER* r;

    fprintf(stderr,"initReceiver %d: scale_factor=%f\n",rx,scale_factor);
    r=&receiver[rx];

    r->id=rx;

    error = cudaHostAlloc(&r->receiverdata, COMPLEX_SIGNAL_SIZE*sizeof(cufftComplex), cudaHostAllocMapped);
    if (error != cudaSuccess) {
       fprintf(stderr, "initReceiver: Error cudaHostAlloc for receiver data %d\n", error);
       exit(EXIT_FAILURE);
    }

    error = cudaHostGetDevicePointer(&(r->deviceReceiverdata), r->receiverdata, 0);
    if (error != cudaSuccess) {
       fprintf(stderr, "initReceiver: Error receiverdata cudaHostGetDevicePointer %d\n", error);
       exit(EXIT_FAILURE);
    }


    //fprintf(stderr,"slice size=%d\n",(COMPLEX_SIGNAL_SIZE/D_SIZE_384K)*sizeof(cufftComplex));
    error = cudaHostAlloc(&r->slice, (COMPLEX_SIGNAL_SIZE/D_SIZE_384K)*sizeof(cufftComplex), cudaHostAllocMapped);
    if (error != cudaSuccess) {
        fprintf(stderr, "processReceiverData: Error cudaHostAlloc for slice data %d\n", error);
        exit(EXIT_FAILURE);
    }

    error = cudaHostGetDevicePointer(&r->deviceSlice, r->slice, 0);
    if (error != cudaSuccess) {
       fprintf(stderr, "processReceiveData: Error slice data cudaHostGetDevicePointer %d\n", error);
       exit(EXIT_FAILURE);
    }


    //fprintf(stderr,"RX_TD_MAXSIZE=%d\n",RX_TD_MAXSIZE);
    //fprintf(stderr,"decimate size=%d\n",(int)(RX_TD_MAXSIZE*sizeof(cufftComplex)));
    error = cudaHostAlloc(&r->decimate, RX_TD_MAXSIZE*sizeof(cufftComplex), cudaHostAllocMapped);
    if (error != cudaSuccess) {
       fprintf(stderr, "processReceiverData: Error cudaHostAlloc for decimate %d\n", error);
       exit(EXIT_FAILURE);
    }

    error = cudaHostGetDevicePointer(&r->deviceDecimate, r->decimate, 0);
    if (error != cudaSuccess) {
       fprintf(stderr, "processReceiveData: Error decimate cudaHostGetDevicePointer %d\n", error);
       exit(EXIT_FAILURE);
    }

    //fprintf(stderr,"tdoutput size=%d\n",(int)(RX_TD_MAXSIZE*sizeof(cufftComplex)));
    error = cudaHostAlloc(&r->tdOutput, RX_TD_MAXSIZE*sizeof(cufftComplex), cudaHostAllocMapped);
    if (error != cudaSuccess) {
        fprintf(stderr, "initHermes: Error cudaHostAlloc for td output data %d\n", error);
        exit(EXIT_FAILURE);
    }

    error = cudaHostGetDevicePointer(&r->deviceTdOutput, r->tdOutput, 0);
    if (error != cudaSuccess) {
        fprintf(stderr, "initHermes: Error td output cudaHostGetDevicePointer %d\n", error);
        exit(EXIT_FAILURE);
    }

    //fprintf(stderr,"output size=%d\n",(int)(RX_TD_MAXSIZE*sizeof(char)*6));
    error = cudaHostAlloc(&r->output, RX_TD_MAXSIZE*sizeof(char)*6, cudaHostAllocMapped);
    if (error != cudaSuccess) {
        fprintf(stderr, "initHermes: Error cudaHostAlloc for output data %d\n", error);
        exit(EXIT_FAILURE);
    }

    error = cudaHostGetDevicePointer(&r->deviceOutput, r->output, 0);
    if (error != cudaSuccess) {
        fprintf(stderr, "initHermes: Error output cudaHostGetDevicePointer %d\n", error);
        exit(EXIT_FAILURE);
    }

    result=sem_init(&r->inputReady, 0, 0);
    if(result!=0) {
        fprintf(stderr,"initReceiver %d: sem_init failed for inputReady%d\n", rx, result);
        exit(EXIT_FAILURE);
    }

    result=sem_init(&r->outputReady, 0, 0);
    if(result!=0) {
        fprintf(stderr,"initReceiver %d: sem_init failed for outputReady%d\n", rx, result);
        exit(EXIT_FAILURE);
    }


    if((result=pthread_create(&r->receiverThreadId, NULL, receiverThread, r)) < 0) {
        fprintf(stderr, "receiverThread create failed %d\n",result);
        exit(EXIT_FAILURE);
    }

}

__global__ void
gpu_mix_and_convolve(const cufftComplex* d_fft, const cufftComplex* d_fir_fft,
                     cufftComplex* d_receiver, const int nrot,
                     const float scale, int d_size)
{
    const size_t numThreads = blockDim.x * gridDim.x;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t new_index;

    for (int i = tid; i < COMPLEX_SIGNAL_SIZE; i += numThreads) {
        new_index = (i >= nrot) ? i - nrot : COMPLEX_SIGNAL_SIZE - nrot + i;

        // Skip computing unneeded bins.
        if (new_index > COMPLEX_SIGNAL_SIZE / d_size)
            continue;

        d_receiver[new_index] = ComplexScale(ComplexMul(d_fft[i], d_fir_fft[new_index]), scale);
    }
}

__global__ void
gpu_decimate(const cufftComplex* deviceReceiver, cufftComplex* deviceSlice, int d_size, int outrot) {
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t new_index;

    new_index = (threadId >= outrot) ? threadId - outrot : COMPLEX_SIGNAL_SIZE - outrot + threadId;
    deviceSlice[threadId] = deviceReceiver[new_index];
}

__global__ void
gpu_ifft_postprocess(const cufftComplex* d_slice, cufftComplex* d_rx_td,
                     char* d_rx_td_24bit, int decimate, int d_size, int rx_td_size,float scale_factor
                    )
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= rx_td_size)
        return;

    int idx = tid * decimate + (P_SIZE-1)/2/d_size;

    d_rx_td[tid]=d_slice[idx];

    // Note: I & Q must be swapped.
    long tempQ = (long)((double)d_rx_td[tid].x * scale_factor); //* SCALE_FACTOR);
    long tempI = (long)((double)d_rx_td[tid].y * scale_factor); //* SCALE_FACTOR);


    // Load samples in big endian format.
    int baseindex = tid * 6;            // start of the 24 bit sample
    d_rx_td_24bit[baseindex++] = (char)((tempI >> 16) & 0xff);
    d_rx_td_24bit[baseindex++] = (char)((tempI >> 8) & 0xff);
    d_rx_td_24bit[baseindex++] = (char)((tempI >> 0) & 0xff);
    d_rx_td_24bit[baseindex++] = (char)((tempQ >> 16) & 0xff);
    d_rx_td_24bit[baseindex++] = (char)((tempQ >> 8) & 0xff);
    d_rx_td_24bit[baseindex++] = (char)((tempQ >> 0) & 0xff);

}

void* receiverThread(void* arg) {
    int result;
    cudaError_t error;
    RECEIVER* r=(RECEIVER*)arg;

#ifdef TIMING
    long long starttime;
    long long endtime;
#endif

    fprintf(stderr,"receiverThread %d: running on cpu %d\n", r->id, sched_getcpu());

    while(1) {
        result=sem_wait(&r->inputReady);
        if(result!=0) {
            fprintf(stderr, "receiverThread: sem_wait failed for inputReady: %d\n", result);
            exit(EXIT_FAILURE);
        }

//fprintf(stderr,"gpu_mix_and_convolve<<<%d,%d>>> rx=%d rotate=%d, scale=%f, d_size=%d\n", COMPLEX_SIGNAL_SIZE/8192,1024,r->id,r->rotate,r->scale,r->d_size);

        gpu_mix_and_convolve<<<COMPLEX_SIGNAL_SIZE/8192, 1024>>>
            (deviceFrequencysamples, r->deviceFilter, r->deviceReceiverdata, r->rotate, r->scale, r->d_size);
/*
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error != cudaSuccess) {
      // print the CUDA error message and exit
      fprintf(stderr,"gpu_mix_and_convolve CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
*/

        //gpu_decimate<<<COMPLEX_SIGNAL_SIZE/1024/r->d_size, 1024>>>
        gpu_decimate<<<COMPLEX_SIGNAL_SIZE/8192, 1024>>>
            (r->deviceReceiverdata, r->deviceSlice, r->d_size, r->outrot );
/*
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error != cudaSuccess) {
      // print the CUDA error message and exit
      fprintf(stderr,"gpu_decimate CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
*/

        // inverse FFT
        cufftResult err=cufftExecC2C(r->planC2C, r->deviceSlice, r->deviceSlice, CUFFT_INVERSE);
        if(err!=CUFFT_SUCCESS) {
           fprintf(stderr,"Error executing planC2C for input buffer: %s\n", _cudaGetErrorEnum(err));
           exit(EXIT_FAILURE);
        }

        // convert to 24 bit samples
        gpu_ifft_postprocess<<<r->rx_td_size/1024 + 1, 1024>>>
            (r->deviceSlice, r->deviceTdOutput, r->deviceOutput, r->ifft_decimate_factor, r->d_size, r->rx_td_size, scale_factor);

        // need to sync as last stage
        cudaDeviceSynchronize();
        error = cudaGetLastError();
        if(error != cudaSuccess) {
          // print the CUDA error message and exit
          fprintf(stderr,"gpu_ifft_postprocess CUDA error: %s\n", cudaGetErrorString(error));
          exit(-1);
        }

        result=sem_post(&r->outputReady);
        if(result!=0) {
            fprintf(stderr, "receiverThread: sem_post failed for outputReady: %d\n", result);
            exit(EXIT_FAILURE);
        }

    }
}
