/**
* @file receiver.cuh
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

#define MAX_RECEIVER 8

extern float scale_factor;

typedef struct _RECEIVER {
    int id;
    int samplingrate;
    int outputrate;
    long frequency;
    int rotate;
    int slicesamples;
    int rx_td_size;
    cufftComplex *filter, *deviceFilter;
    cufftComplex *receiverdata, *deviceReceiverdata;
    cufftComplex *slice, *deviceSlice;
    cufftComplex *decimate, *deviceDecimate;
    cufftComplex *tdOutput, *deviceTdOutput;
    char *output, *deviceOutput;
    sem_t inputReady;
    sem_t outputReady;
    pthread_t receiverThreadId;
    float scale;
    int d_size;
    int d_size_2;
    int ifft_decimate_factor;
    int outrot;
    cufftHandle planC2C;
} RECEIVER;

extern RECEIVER receiver[MAX_RECEIVER];

void initReceiver(int rx);
