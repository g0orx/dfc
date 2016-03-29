/**
* @file inputbuffer.cu
* @brief inputbuffer and semaphores
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
