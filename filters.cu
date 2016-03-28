#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <helper_cuda.h>

#include "common.cuh"
#include "filters.cuh"
#include "coeff.cuh"

cufftComplex* filter48k;
cufftComplex* filter96k;
cufftComplex* filter192k;
cufftComplex* filter384k;

cufftComplex* deviceFilter48k;
cufftComplex* deviceFilter96k;
cufftComplex* deviceFilter192k;
cufftComplex* deviceFilter384k;

//static cufftHandle      planR2C;

__global__ void	gpu_make_analytic(cufftComplex*);

void generateFilter(double samplingrate,double Fb,cufftComplex* deviceFilter,cufftComplex* filter);

void loadFilters(int pcie) {
    
fprintf(stderr,"filter size=%ld\n",(COMPLEX_SIGNAL_SIZE+1)*sizeof(cufftComplex));

    cudaError_t result = cudaHostAlloc(&filter48k, (COMPLEX_SIGNAL_SIZE+1)*sizeof(cufftComplex), cudaHostAllocMapped);
    if (result != cudaSuccess) {
       fprintf(stderr, "Error filter 48k cudaHostAlloc %d\n", result);
       exit(EXIT_FAILURE);
    }
    result = cudaHostGetDevicePointer(&deviceFilter48k, filter48k, 0);
    if (result != cudaSuccess) {
       fprintf(stderr, "Error filter 48k cudaHostGetDevicePointer %d\n", result);
       exit(EXIT_FAILURE);
    }
    generateFilter(samplingrate,48000.0,deviceFilter48k,filter48k);

    result = cudaHostAlloc(&filter96k, (COMPLEX_SIGNAL_SIZE+1)*sizeof(cufftComplex), cudaHostAllocMapped);
    if (result != cudaSuccess) {
       fprintf(stderr, "Error filter 96k cudaHostAlloc %d\n", result);
       exit(EXIT_FAILURE);
    }
    result = cudaHostGetDevicePointer(&deviceFilter96k, filter96k, 0);
    if (result != cudaSuccess) {
       fprintf(stderr, "Error filter 96k cudaHostGetDevicePointer %d\n", result);
       exit(EXIT_FAILURE);
    }
    generateFilter(samplingrate,96000.0,deviceFilter96k,filter96k);

    result = cudaHostAlloc(&filter192k, (COMPLEX_SIGNAL_SIZE+1)*sizeof(cufftComplex), cudaHostAllocMapped);
    if (result != cudaSuccess) {
       fprintf(stderr, "Error filter 192k cudaHostAlloc %d\n", result);
       exit(EXIT_FAILURE);
    }
    result = cudaHostGetDevicePointer(&deviceFilter192k, filter192k, 0);
    if (result != cudaSuccess) {
       fprintf(stderr, "Error filter 192k cudaHostGetDevicePointer %d\n", result);
       exit(EXIT_FAILURE);
    }
    generateFilter(samplingrate,192000.0,deviceFilter192k,filter192k);

    result = cudaHostAlloc(&filter384k, (COMPLEX_SIGNAL_SIZE+1)*sizeof(cufftComplex), cudaHostAllocMapped);
    if (result != cudaSuccess) {
       fprintf(stderr, "Error filter 384k cudaHostAlloc %d\n", result);
       exit(EXIT_FAILURE);
    }
    result = cudaHostGetDevicePointer(&deviceFilter384k, filter384k, 0);
    if (result != cudaSuccess) {
       fprintf(stderr, "Error filter 384k cudaHostGetDevicePointer %d\n", result);
       exit(EXIT_FAILURE);
    }
    generateFilter(samplingrate,384000.0,deviceFilter384k,filter384k);
}

void generateFilter(double samplingrate,double Fb,cufftComplex* deviceFilter,cufftComplex* filter) {

    cufftReal* coeff;
    cufftReal* deviceCoeff;
    cufftHandle planR2C;

fprintf(stderr,"coeffs size=%ld\n",DFT_BLOCK_SIZE*sizeof(cufftReal));
    cudaError_t result = cudaHostAlloc(&coeff, DFT_BLOCK_SIZE*sizeof(cufftReal), cudaHostAllocMapped);
    if (result != cudaSuccess) {
       fprintf(stderr, "Error coeff cudaHostAlloc %d\n", result);
       exit(EXIT_FAILURE);
    }

    result = cudaHostGetDevicePointer(&deviceCoeff, coeff, 0);
    if (result != cudaSuccess) {
       fprintf(stderr, "Error coeff cudaHostGetDevicePointer %d\n", result);
       exit(EXIT_FAILURE);
    }

    double *H=calcFilter(samplingrate,0.0,Fb,P_SIZE,25.0);
    for(int i=0;i<P_SIZE;i++) {
        coeff[i]=(cufftReal)H[i];
    }
    free(H);

    for(int i=P_SIZE; i<DFT_BLOCK_SIZE;i++) {
        coeff[i]=0.0;
    }

#ifdef DUMPDATA
for(int i=0;i<1024;i++) {
    fprintf(stderr,"coeff %d=%.24f\n", i, coeff[i]);
}
#endif

    cufftResult error = cufftPlan1d(&planR2C, DFT_BLOCK_SIZE, CUFFT_R2C, 1);
    if(error!=CUFFT_SUCCESS) {
       fprintf(stderr,"Error creating cufftPlan1d for FIR: %s\n", _cudaGetErrorEnum(error));
       exit(EXIT_FAILURE);
    }

    error = cufftExecR2C(planR2C, deviceCoeff, deviceFilter);
    if (error != CUFFT_SUCCESS) {
       fprintf(stderr, "Error cufftExecR2C (planR2C) %d\n", result);
       exit(EXIT_FAILURE);
    }

    gpu_make_analytic<<<COMPLEX_SIGNAL_SIZE/4096, 1024>>>(deviceFilter);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
      // print the CUDA error message and exit
      fprintf(stderr,"gpu_make_analytic error: %s\n", cudaGetErrorString(err));
      exit(-1);
    }


#ifdef DUMPDATA
for(int i=0;i<1024;i++) {
    fprintf(stderr,"analytic %d=%.24f:%.24f\n", i, filter[i].x, filter[i].y);
}
#endif

    error = cufftDestroy(planR2C);
    if (error != CUFFT_SUCCESS) {
       fprintf(stderr, "Error cufftDestroy (planR2C) %d\n", result);
       exit(EXIT_FAILURE);
    }

    result = cudaFreeHost(coeff);
    if (result != cudaSuccess) {
       fprintf(stderr, "Error coeff cudaFreeHost %d\n", result);
       exit(EXIT_FAILURE);
    }
}

cufftComplex* getFilter(int rate) {
    cufftComplex* filter;
    filter=(cufftComplex*)0;
    switch(rate) {
        case 0:
           filter=filter48k;
           break;
        case 1:
           filter=filter96k;
           break;
        case 2:
           filter=filter192k;
           break;
        case 3:
           filter=filter384k;
           break;
        default:
           fprintf(stderr,"getFilter: invalid rate %d/n", rate);
           exit(EXIT_FAILURE);
           break;
    }
    return filter;
}

cufftComplex* getDeviceFilter(int rate) {
    cufftComplex* filter;
    filter=(cufftComplex*)0;
    switch(rate) {
        case 0:
           filter=deviceFilter48k;
           break;
        case 1:
           filter=deviceFilter96k;
           break;
        case 2:
           filter=deviceFilter192k;
           break;
        case 3:
           filter=deviceFilter384k;
           break;
        default:
           fprintf(stderr,"getDeviceFilter: invalid rate %d/n", rate);
           exit(EXIT_FAILURE);
           break;
    }
    return filter;
}

