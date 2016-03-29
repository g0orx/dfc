/**
* @file pcie.cu
* @brief Read samples from PCIe interface
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

#include "common.cuh"
#include "inputbuffer.cuh"
#include "pcie.cuh"
#include "time.cuh"

static int xillyFd;
static pthread_t pcieThread;

void* pcieLoop(void* args);

void initPcie() {

    fprintf(stderr, "init_pcie\n");
    xillyFd=open("/dev/xillybus_read_32", O_RDONLY);
    if(xillyFd<0) {
        perror("Failed to open xillybus_read_32");
        exit(EXIT_FAILURE);
    }

    int result=pthread_create(&pcieThread, NULL, pcieLoop, NULL);
    if(result<0) {
        fprintf(stderr, "Error creating PCIe thread: %d\n", result);
        exit(EXIT_FAILURE);
    }
}

void* pcieLoop(void* args) {
    int bytestoread=L_SIZE*sizeof(short);
    int count;
    int offset;
    int result;

#ifdef TIMING
    long long starttime;
    long long endtime;
    int reads;
#endif
    fprintf(stderr,"pcieLoop: running on cpu %d\n", sched_getcpu());
    while(1) {
        result=sem_wait(&inputBufferEmpty);
        if(result!=0) {
            fprintf(stderr,"pcieLoop: sem_wait failed for inputBufferEmpty %d\n", result);
            exit(EXIT_FAILURE);
        }
#ifdef TIMING
        starttime=current_timestamp();
        reads=0;
#endif
        offset=0;
        while(offset!=bytestoread) {
            count=read(xillyFd, &inputBuffer[offset/sizeof(short)], bytestoread-offset);
            if(count<=0) {
                if(errno != EINTR) {
                    fprintf(stderr, "pcieLooperror error reading: %s\n", strerror(errno));
                    exit(EXIT_FAILURE);
                }
            } else {
                offset=offset+count;
            }
#ifdef TIMING
            reads++;
#endif
        }
#ifdef TIMING
        endtime=current_timestamp();
        fprintf(stderr, "%d in %lld ms %d reads\n", L_SIZE, endtime-starttime, reads);
#endif
        // signal input buffer is ready
        result=sem_post(&inputBufferFull);
        if(result!=0) {
            fprintf(stderr,"pcieLoop: sem_post failed for inputBufferFull %d\n", result);
            exit(EXIT_FAILURE);
        }
    }
}

