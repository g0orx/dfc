/**
* @file main.cu
* @brief main code - starts here
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

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <pthread.h>
#include <semaphore.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include "common.cuh"
#include "filters.cuh"
#include "inputbuffer.cuh"
#include "rawbuffer.cuh"
#include "pcie.cuh"
#include "socket.cuh"
#include "file.cuh"
#include "hermes.cuh"
#include "dfc.cuh"
#include "start.cuh"
#include "audio.cuh"
#include "receiver.cuh"

struct cudaDeviceProp deviceProp;
int driverVersion;
int runtimeVersion;

int source=SOURCE_PCIE;

int samplingrate;
float hzperbin;

int adc=1;
int capture=0;

sem_t finish;

char interface[8];

void processArgs(int,char*[]);

void intHandler(int sig) {
    fprintf(stderr,"got signal %d\n",sig);
    dfcTerminate();
    exit(0);
}

void checkCUDA() {
    int devices = 0;
    cudaError_t error_id = cudaGetDeviceCount(&devices);
    if (error_id != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDeviceCount returned %d: %s\n",
                (int)error_id, cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    fprintf(stderr, "Found %d CUDA devices\n", devices);
    if(devices!=1) {
        fprintf(stderr, "Currently ony works with 1 devices\n");
        exit(EXIT_FAILURE);
    }

    cudaSetDevice(0);
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    fprintf(stderr, "driver=%d runtime=%d\n", driverVersion, runtimeVersion);
    if(!deviceProp.integrated) {
        fprintf(stderr,"Currently only works with integrated devices (Jetson)\n");
        exit(EXIT_FAILURE);
    }

    fprintf(stderr,"compute=%d.%d\n",deviceProp.major,deviceProp.minor);
    fprintf(stderr,"max threadsPerBlock=%d\n",deviceProp.maxThreadsPerBlock);
    fprintf(stderr,"max regsPerSM=%d\n",deviceProp.regsPerMultiprocessor);

}

int main(int args,char* argv[]) {

    strcpy(interface,"eth0");

    processArgs(args,argv);

    checkCUDA();

    // get the audio stream started
    if(audio) {
        audio_init();
    }

    switch(source) {
        case SOURCE_PCIE:
            fprintf(stderr,"hermes: data source: PCIe\n");
            samplingrate=PCIE_SAMPLE_RATE;
            break;
        case SOURCE_SOCKET:
            fprintf(stderr,"hermes: data source: Socket %s\n",interface);
            samplingrate=SOCKET_SAMPLE_RATE;
            break;
        case SOURCE_FILE:
            fprintf(stderr,"hermes: data source: File raw.bin\n");
            samplingrate=SOCKET_SAMPLE_RATE;
            break;
    }

    hzperbin=(float)samplingrate/2.0F/(float)COMPLEX_SIGNAL_SIZE;
    
    fprintf(stderr,"samplingrate=%d hzperbin=%f COMPLEX_SIGNAL_SIZE=%d\n",samplingrate, hzperbin, COMPLEX_SIGNAL_SIZE);

    loadFilters(source==SOURCE_PCIE);
    initInputBuffer();
    initRawBuffer();

    switch(source) {
        case SOURCE_PCIE:
            initPcie();
            break;
        case SOURCE_SOCKET:
            sendStart(adc);
            initSocket();
            break;
        case SOURCE_FILE:
            initFile();
            break;
    }

    initDfc();

    initHermes();

    signal(SIGKILL, intHandler);
    signal(SIGINT, intHandler);


    // wait for semaphore
    int result=sem_init(&finish, 0, 0);
    if(result!=0) {
        fprintf(stderr,"hermes: sem_init failed %d\n", result);
        exit(EXIT_FAILURE);
    }

    result=sem_wait(&finish);
    if(result!=0) {
        fprintf(stderr,"hermes: sem_wait failed %d\n", result);
        exit(EXIT_FAILURE);
    }
    
    fprintf(stderr, "hermes: ending\n");
    
    return EXIT_SUCCESS;
}

void processArgs(int args,char* argv[]) {
    int i;
    int option;
    struct option options[] = {
        {"pcie",   no_argument, 0, 0 },
        {"socket", optional_argument, 0, 0 },
        {"adc", required_argument, 0, 0 },
        {"audio", required_argument, 0, 0 },
        {"audio_buffer", required_argument, 0, 0 },
        {"capture", no_argument, 0, 0 },
        {"file", no_argument, 0, 0 },
        {"scale", required_argument, 0, 0 },
        {"id", required_argument, 0, 0 },
        {0,0,0,0}
    };
    while(1) {
        if((i=getopt_long(args,argv,"ps::a::cf",options,&option))==-1) break;
        switch(i) {
            case 0:
                switch(option) {
                   case 0:
                        source=SOURCE_PCIE; 
                        break;
                   case 1:
                        source=SOURCE_SOCKET;
                        if(optarg!=0) {
                            strcpy(interface,optarg);
                        }
                        break;
                   case 2:
                        adc=0;
                        if(optarg!=0) {
                            adc=atoi(optarg);
                        }
                        break;
                   case 3:
                        audio=false;
                        if(optarg!=0) {
                            audio=atoi(optarg);
                        }
                        break;
                   case 4:
                        if(optarg!=0) {
                            audio_buffer_size=atoi(optarg);
                        }
                        break;
                   case 5:
                        capture=1;
                        break;
                   case 6:
                        source=SOURCE_FILE;
                        break;
                   case 7:
                        if(optarg!=0) {
                            scale_factor=atof(optarg);
                        }
                        break;
                   case 8:
                        if(optarg!=0) {
                            hpsdr_id=atoi(optarg);
                        }
                        break;
                 }
                 break;
            case 'p':
                 source=SOURCE_PCIE; 
                 break;
            case 's':
                 source=SOURCE_SOCKET; 
                 if(optarg!=0) {
                     strcpy(interface,optarg);
                 }
                 break;
            case 'a':
                 adc=0; 
                 if(optarg!=0) {
                     adc=atoi(optarg);
                 }
                 break;
            case 'c':
                 capture=1; 
                 break;
            case 'f':
                 source=SOURCE_FILE;
                 break;
            default:
                 fprintf(stderr,"Invalid option -%c\n",i);
                 exit(EXIT_FAILURE);
        }
    }
}
