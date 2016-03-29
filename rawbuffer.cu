/**
* @file rawbuffer.cu
* @brief rawbuffer for bandscope data
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
