/**
* @file hermes.cu
* @brief Hermes emulation
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
#include "dfc.cuh"
#include "inputbuffer.cuh"
#include "rawbuffer.cuh"
#include "filters.cuh"
#include "receiver.cuh"
#include "hermes.cuh"
#include "time.cuh"
#include "audio.cuh"

#define PORT 1024
#define MAX_BUFFER_LEN 1032

#define HERMES_FW_VERSION 26
#define HERMES_ID 0x01
#define HERMES_LITE_ID 0x06

#define SYN 0x7f



//#define SCALE_FACTOR  0x7fffffffL
#define SCALE_FACTOR  8388607.0 // 2^24-1

int hpsdr_id=HERMES_ID;

static int slicesamples;

static pthread_t readThreadId;
static pthread_t processThreadId;
static pthread_t processRawThreadId;

static int hermesSocket;

static unsigned char hw_address[6];

static int state=0; // 0 = idle, 1 = running
struct sockaddr_in clientAddr;

static int sendIQ=0;
static int sendRaw=0;

static int outputrate=-1; // nothing
static int outputsamplerate=0;
static int receivers=1;
static int mox=0;
static int commonfrequency=0;

static long tx_sequence=0;
static long raw_sequence=0;

#define MAX_RECEIVERS 7
/*
static long frequency[MAX_RECEIVERS] = {14150000,14150000,14150000,14150000,14150000,14150000,14150000};
static int rotate[MAX_RECEIVERS] = {0,0,0,0,0,0,0};
static cufftComplex* receiverdata[MAX_RECEIVERS];
static cufftComplex* deviceReceiverdata[MAX_RECEIVERS];
static cufftComplex* slicedata[MAX_RECEIVERS];
static cufftComplex* deviceSlicedata[MAX_RECEIVERS];
static cufftComplex* slice[MAX_RECEIVERS];
static cufftComplex* deviceSlice[MAX_RECEIVERS];
static cufftComplex* decimate[MAX_RECEIVERS];
static cufftComplex* deviceDecimate[MAX_RECEIVERS];
*/

static cufftComplex* filter;
static cufftComplex* deviceFilter;

static char* output[MAX_RECEIVERS];
static char* deviceOutput[MAX_RECEIVERS];

static float scale;
static int d_size;
static int d_size_2;
static int ifft_decimate_factor;
static int outrot;

static cufftHandle planC2C;

#define FRAME_LENGTH 1032
static unsigned char frame[FRAME_LENGTH];
static int frameoffset;
static unsigned char rawframe[FRAME_LENGTH];
static int rawframeoffset;

void* readThread(void* arg);
void* processThread(void* arg);
void* processRawThread(void* arg);
void processClientData(unsigned char* buffer);
void processClientFrame(unsigned char* buffer);

void initHermes() {
    int result;
    cudaError_t error;

    fprintf(stderr,"initHermes\n");

    scale=1.0;

    for(int i=0;i<FRAME_LENGTH;i++) {
        frame[i]='\0';
    }

    frame[0]=0xef;
    frame[1]=0xfe;
    frame[2]=0x01;
    frame[3]=0x06;
    frame[4]=0x00;
    frame[5]=0x00;
    frame[6]=0x00;
    frame[7]=0x00;

    frame[8]=0x7f;
    frame[9]=0x7f;
    frame[10]=0x7f;
    frame[11]=0x00;
    frame[12]=0x1e;
    frame[13]=0x00;
    frame[14]=0x00;
    frame[15]=HERMES_FW_VERSION;

    frame[520]=0x7f;
    frame[521]=0x7f;
    frame[522]=0x7f;
    frame[523]=0x00;
    frame[524]=0x1e;
    frame[525]=0x00;
    frame[526]=0x00;
    frame[527]=HERMES_FW_VERSION;

    frameoffset=16;

    rawframe[0]=0xef;
    rawframe[1]=0xfe;
    rawframe[2]=0x01;
    rawframe[3]=0x04;
    rawframe[4]=0x00;
    rawframe[5]=0x00;
    rawframe[6]=0x00;
    rawframe[7]=0x00;
    rawframeoffset=8;

    if((result=pthread_create(&readThreadId, NULL, readThread, NULL)) < 0) {
        fprintf(stderr, "readThread create failed %d\n",result);
        exit(EXIT_FAILURE);
    }

    if((result=pthread_create(&processThreadId, NULL, processThread, NULL)) < 0) {
        fprintf(stderr, "processThread create failed %d\n",result);
        exit(EXIT_FAILURE);
    }

    if((result=pthread_create(&processRawThreadId, NULL, processRawThread, NULL)) < 0) {
        fprintf(stderr, "processRawThread create failed %d\n",result);
        exit(EXIT_FAILURE);
    }

}

void* readThread(void* arg) {
    struct sockaddr_in readAddr;
    uint8_t readBuffer[MAX_BUFFER_LEN];
    socklen_t readLength;
    struct ifreq ifr;

    readLength = sizeof(readAddr);

    fprintf(stderr,"hermes readThread: running on cpu %d\n", sched_getcpu());

    hermesSocket = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (hermesSocket < 0) {
        perror("readThread: create socket failed for hermesSocket\n");
        exit(EXIT_FAILURE);
    }

    int on=1;
    int rc = setsockopt(hermesSocket, SOL_SOCKET, SO_REUSEADDR, (const void*)&on, sizeof(on));
    if (rc != 0) {
        fprintf(stderr, "readThread: cannot set SO_REUSEADDR: rc=%d\n", rc);
        exit(EXIT_FAILURE);
    }

    // Bind to this interface.
    readAddr.sin_family = AF_INET;
    readAddr.sin_port = htons(PORT);
    readAddr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(hermesSocket, (struct sockaddr*) &readAddr, sizeof(readAddr)) < 0) {
        perror("readThread: bind socket failed for hermesSocket\n");
        exit(EXIT_FAILURE);
    }

    // Allow broadcast on the socket.
    rc = setsockopt(hermesSocket, SOL_SOCKET, SO_BROADCAST, (const void*)&on, sizeof(on));
    if (rc != 0) {
        fprintf(stderr, "readThread: cannot set SO_BROADCAST: rc=%d\n", rc);
        exit(EXIT_FAILURE);
    }

    ifr.ifr_addr.sa_family = AF_INET;
    strncpy(ifr.ifr_name, interface, IFNAMSIZ-1);
    ioctl(hermesSocket, SIOCGIFADDR, &ifr);

    unsigned char* u = (unsigned char*)&ifr.ifr_addr.sa_data;
    for (int k = 0; k < 6; k++) hw_address[k] = u[k];

    fprintf(stderr, "readThread: listening on %s (%02x:%02x:%02x:%02x:%02x:%02x)\n",
           inet_ntoa(((struct sockaddr_in *)&ifr.ifr_addr)->sin_addr),
           hw_address[0], hw_address[1], hw_address[2],
           hw_address[3], hw_address[4], hw_address[5]);

    unsigned char discoverBuffer[MAX_BUFFER_LEN] =
           { 0xef, 0xfe, 0x02, 0, 0, 0, 0, 0, 0, HERMES_FW_VERSION, hpsdr_id };

    while(1) {

        if ((rc=recvfrom(hermesSocket, readBuffer, sizeof(readBuffer), 0,
                      (struct sockaddr*)&readAddr, &readLength)) < 0) {
            fprintf(stderr, "readThread: Bad recvfrom %d", rc);
            exit(EXIT_FAILURE);
        }

        //fprintf(stderr,"recvfrom: %d bytes\n", rc);

        if ((readBuffer[0] == 0xef) && (readBuffer[1] == 0xfe))  {
            switch(readBuffer[2]) {
                case 1:
                    // data
                    if(state) {
                        // check if from expected client
                        if(memcmp(&clientAddr,&readAddr, readLength)==0) {
                            processClientData(readBuffer);
                        } else {
                            // ignore
                        }
                    } else {
                        processClientData(readBuffer);
                    }
                    break;
                case 2:
                    fprintf(stderr, "readThread: received discovery from %s %d\n",
                        inet_ntoa(readAddr.sin_addr), htons(readAddr.sin_port));
                    for (int i = 0; i < 6; i++) {
                        discoverBuffer[3 + i] = hw_address[i];
                    }
                    discoverBuffer[2] |= state;
                    discoverBuffer[10]=hpsdr_id;
                    for (int i = 11; i < 60; i++)
                        discoverBuffer[i] = 0;
                    if ((rc=sendto(hermesSocket, discoverBuffer, 60, 0,
                               (struct sockaddr*)&readAddr, sizeof(readAddr))) < 0) {
                        fprintf(stderr, "readThread: Bad sendto %d",rc);
                        exit(EXIT_FAILURE);
                    }
                   break;
               case 4:
                   // start/stop command
                   switch(readBuffer[3]) {
                       case 0:
                           if(state==0)  {
                               fprintf(stderr,"readThread: ignoring stop command from %s\n",
                                   inet_ntoa(readAddr.sin_addr));
                           } else if(memcmp(&clientAddr,&readAddr, readLength)==0) {
                               state=0;
                               sendIQ=0;
                               sendRaw=0;
                               tx_sequence=0;
                               raw_sequence=0;
                           } else {
                               fprintf(stderr,"readThread: ignoring stop command from %s\n",
                                   inet_ntoa(readAddr.sin_addr));
                           }
                           break;
                       case 1:
                           if(state==0)  {
                               memcpy(&clientAddr,&readAddr, readLength);
                               state=1;
                               sendIQ=1;
                               sendRaw=0;
                           } else if(memcmp(&clientAddr,&readAddr, readLength)==0) {
                               sendIQ=1;
                               sendRaw=0;
                           } else {
                               fprintf(stderr,"readThread: ignoring start command %d from %s\n",
                                   readBuffer[3], inet_ntoa(readAddr.sin_addr));
                           }
                           break;
                       case 2:
                           if(state==0)  {
                               memcpy(&clientAddr,&readAddr, readLength);
                               state=1;
                               sendIQ=0;
                               sendRaw=1;
                           } else if(memcmp(&clientAddr,&readAddr, readLength)==0) {
                               sendIQ=0;
                               sendRaw=1;
                           } else {
                               fprintf(stderr,"readThread: ignoring start command %d from %s\n",
                                   readBuffer[3], inet_ntoa(readAddr.sin_addr));
                           }
                           break;
                       case 3:
                           if(state==0)  {
                               memcpy(&clientAddr,&readAddr, readLength);
                               state=1;
                               sendIQ=1;
                               sendRaw=1;
                           } else if(memcmp(&clientAddr,&readAddr, readLength)==0) {
                               sendIQ=1;
                               sendRaw=1;
                           } else {
                               fprintf(stderr,"readThread: ignoring start command %d from %s\n",
                                   readBuffer[3], inet_ntoa(readAddr.sin_addr));
                           }
                           break;
                   }
                   fprintf(stderr,"readThread: received start/stop command: state=%d sendIQ=%d sendRaw=%d\n",
                           state, sendIQ, sendRaw);
                   break;
                default:
                   break;
            }
        } else {
            fprintf(stderr, "readThread: unexpected packet from %s (0x%02x 0x%02x 0x%02x)\n",
                inet_ntoa(readAddr.sin_addr),
                readBuffer[0], readBuffer[1], readBuffer[2]);
        }
    }
    
}

void processClientData(unsigned char* buffer) {
    int ep=buffer[3]&0xFF;
    if(ep==2) {
        processClientFrame(&buffer[8]);
        processClientFrame(&buffer[520]);
    } else {
        fprintf(stderr,"processClientData: unexpected endpoint %d\n", ep);
    }
}


void processClientFrame(unsigned char* buffer) {
    int id;
    int rate;
    int rcvrs;
    int rx;
    long f;
    int rot;
    cudaError_t error;
    cufftResult cufftError;

    if(buffer[0]==SYN && buffer[1]==SYN && buffer[2]==SYN) {
        mox=buffer[3]&0x01;
        id=(buffer[3]&0xFF)>>1;
        switch(id) {
            case 0:
                rate=buffer[4]&0x03;
                if(rate!=outputrate) {
                    outputrate=rate;
                    switch(rate) {
                        case 0:
                            outputsamplerate=48000;
                            break;
                        case 1:
                            outputsamplerate=96000;
                            break;
                        case 2:
                            outputsamplerate=192000;
                            break;
                        case 3:
                            outputsamplerate=384000;
                            break;
                    }
                    fprintf(stderr,"outputsamplerate=%d\n",outputsamplerate);

                    filter=getFilter(rate);
                    deviceFilter=getDeviceFilter(rate);

                    slicesamples=(int)((float)outputsamplerate/hzperbin);


                    //if(source=SOURCE_PCIE) {
                    //    d_size=256;
                    //    d_size_2=(samplingrate/256)/outputsamplerate;
                    //} else {
                        d_size=(samplingrate/10)/outputsamplerate;
                    //}

                    fprintf(stderr,"d_size=%d\n",d_size);

                    ifft_decimate_factor = (samplingrate / d_size / 2 / outputsamplerate);
                    fprintf(stderr,"ifft_decimate_factor=%d\n",ifft_decimate_factor);

                    outrot = (int)(round((outputsamplerate/2) * NFACTOR) * V_SIZE)+9;
                    fprintf(stderr,"outrot=%d\n",outrot);

                      
                    fprintf(stderr,"planC2C=%d\n",COMPLEX_SIGNAL_SIZE/d_size);
                    cufftError = cufftPlan1d(&planC2C, COMPLEX_SIGNAL_SIZE/d_size, CUFFT_C2C, 1);
                    if(cufftError!=CUFFT_SUCCESS) {
                        fprintf(stderr,"processClientFrame: Error creating cufftPlan1d for Inverse FFT: %s\n", _cudaGetErrorEnum(cufftError));
                        exit(EXIT_FAILURE);
                    }


fprintf(stderr,"P_SIZE:%d V_SIZE:%d L_SIZE:%d RX_TD_SIZE=%d\n", P_SIZE,V_SIZE,L_SIZE,RX_TD_SIZE);
fprintf(stderr,"DFT_BLOCK_SIZE:%d COMPLEX_SIGNAL_SIZE:%d\n", DFT_BLOCK_SIZE,COMPLEX_SIGNAL_SIZE);

                    for(int i=0;i<MAX_RECEIVER;i++) {
                        RECEIVER* r=&receiver[i];
                        r->outputrate=outputsamplerate;
                        r->filter=filter;
                        r->deviceFilter=deviceFilter;
                        r->slicesamples=slicesamples;
                        r->d_size=d_size;
                        r->ifft_decimate_factor=ifft_decimate_factor;
                        r->rx_td_size=RX_TD_SIZE;
                        r->planC2C=planC2C;
                        r->scale=1.0F;
                        r->outrot=outrot;
                        initReceiver(i);
                     }

                }
                rcvrs=((buffer[7]>>3)&0x07)+1;
                if(receivers!=rcvrs) {
                    receivers=rcvrs;
                    fprintf(stderr,"processClientFrame: setting receivers to %d\n", receivers);
                }

                commonfrequency=(buffer[7]>>7)&0x01;
                break;
            case 1: // tx frequency
                break;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                rx=id-2;
                f = (buffer[4]&0xFF) << 24 | (buffer[5]&0xFF) << 16 | (buffer[6]&0xFF) << 8 | (buffer[7]&0xFF);
                rot=(int)((((float)f-((float)outputsamplerate/2.0f))/hzperbin)+0.5f);
                //rot=(int)((((float)f-((float)outputsamplerate/2.0f))/hzperbin));
                //rot=(int)(((float)f/hzperbin)+0.5f);
                if(commonfrequency) {
                    for(rx=0;rx<receivers;rx++) {
                        receiver[rx].frequency=f;
                        receiver[rx].rotate=rot;
                    }
                } else {
                    receiver[rx].frequency=f;
                    receiver[rx].rotate=rot;
                }

                //fprintf(stderr,"set new frequency(%d) %ld rot=%d\n", rx, f, rot);
                break;
            case 9:
            case 10:
            case 11:
            case 12:
            case 13:
            case 14:
            case 15:
            case 16:
                break;
            default:
                break;
        }

        if(audio) {
            audio_write(buffer);
        }
    } else {
        fprintf(stderr,"processClientFrame: syn error 0x%02x 0x%02x 0x%02x\n", 
                buffer[0], buffer[1],buffer[2]);
    }
}

void* processThread(void* arg) {
    int result;
    cudaError_t error;

#ifdef TIMING
    long long starttime;
    long long endtime;
#endif

    fprintf(stderr,"hermes processThread: running on cpu %d\n", sched_getcpu());

    // get the next buffer
    result=sem_post(&frequencyBufferEmpty);
    if(result!=0) {
        fprintf(stderr, "processThread: sem_post failed for frequencyBufferEmpty: %d\n", result);
        exit(EXIT_FAILURE);
    }

    while(1) {

        result=sem_wait(&frequencyBufferFull);
        if(result!=0) {
            fprintf(stderr, "processThread: sem_wait failed for frequencyBufferFull: %d\n", result);
            exit(EXIT_FAILURE);
        }

        if(state && sendIQ) {
#ifdef TIMING
            starttime=current_timestamp();
#endif
            // process the buffer for each receiver
            // TODO handle commonfrequency
            for(int i=0;i<receivers;i++) {
                result=sem_post(&receiver[i].inputReady);
                if(result!=0) {
                    fprintf(stderr, "processRawThread: sem_post failed for inputReady %d: %d\n", i, result);
                    exit(EXIT_FAILURE);
                }
            }

            for(int i=0;i<receivers;i++) {
                result=sem_wait(&receiver[i].outputReady);
                if(result!=0) {
                    fprintf(stderr, "processRawThread: sem_wait failed for inputReady %d: %d\n", i, result);
                    exit(EXIT_FAILURE);
                }
            }
  
            // can get the next buffer
            result=sem_post(&frequencyBufferEmpty);
            if(result!=0) {
                fprintf(stderr, "processThread: sem_post failed for frequencyBufferEmpty: %d\n", result);
                exit(EXIT_FAILURE);
            }

            // copy the IQ samples
//fprintf(stderr,"copying %d IQ samples\n", RX_TD_SIZE);
            for(int i=0;i<RX_TD_SIZE;i++) {
                // I/Q samples for each receiver
                for(int r=0;r<receivers;r++) {
                    for(int j=0;j<6;j++) {
                        frame[frameoffset++]=receiver[r].output[(i*6)+j];
                    }
                }
                // mic samples
                frame[frameoffset++]=0x00;
                frame[frameoffset++]=0x00;

                if(frameoffset<=520) {
                    if(frameoffset+(receivers*6)+2>520) {
//fprintf(stderr,"frameoffset=%d setting to 528\n",frameoffset);
                        frameoffset=528;
                    }
                } else if(frameoffset<=1032) {
                    if(frameoffset+(receivers*6)+2>1032) {
//fprintf(stderr,"frameoffset=%d sendign and setting to 16\n",frameoffset);
                        // send the frame
                        frame[4] = (tx_sequence >> 24) & 0xff;
                        frame[5] = (tx_sequence >> 16) & 0xff;
                        frame[6] = (tx_sequence >> 8) & 0xff;
                        frame[7] = tx_sequence & 0xff;


//fprintf(stderr,"send frame offset=%d seq=%ld\n",frameoffset,tx_sequence);
                        if ((result=sendto(hermesSocket, frame, 1032, 0,
                               (struct sockaddr*)&clientAddr, sizeof(clientAddr))) < 0) {
                            fprintf(stderr, "Error sending data to client %d\n", result);
                            exit(EXIT_FAILURE);
                        }

                        tx_sequence++;
                        frameoffset=16;
                    }
                }
            }

//fprintf(stderr,"copied samples: frameoffset=%d\n",frameoffset);
#ifdef TIMING
            endtime=current_timestamp();
            fprintf(stderr,"process took %lld ms to process %d receivers\n", endtime-starttime, receivers);
#endif

        } else {
            // can get the next buffer
            result=sem_post(&frequencyBufferEmpty);
            if(result!=0) {
                fprintf(stderr, "processThread: sem_post failed for frequencyBufferEmpty: %d\n", result);
                exit(EXIT_FAILURE);
            }

        }

    }
}

void* processRawThread(void* arg) {
    int result;
    fprintf(stderr,"hermes processRawThread: running on cpu %d\n", sched_getcpu());
    while(1) {
        // get the next buffer
        result=sem_post(&rawBufferEmpty);
        if(result!=0) {
            fprintf(stderr, "processRawThread: sem_post failed for rawBufferEmpty: %d\n", result);
            exit(EXIT_FAILURE);
        }

        result=sem_wait(&rawBufferFull);
        if(result!=0) {
            fprintf(stderr, "processRawThread: sem_wait failed for rawBufferFull: %d\n", result);
            exit(EXIT_FAILURE);
        }

        if(state && sendRaw) {
            for(int i=0;i<RAW_BUFFER_SIZE;i++) {
                rawframe[rawframeoffset++]=rawBuffer[i]&0xFF;
                rawframe[rawframeoffset++]=(rawBuffer[i]>>8)&0xFF;
                if(rawframeoffset>=1032) {

                    rawframe[4] = (raw_sequence >> 24) & 0xff;
                    rawframe[5] = (raw_sequence >> 16) & 0xff;
                    rawframe[6] = (raw_sequence >> 8) & 0xff;
                    rawframe[7] = raw_sequence & 0xff;

                    if ((result=sendto(hermesSocket, rawframe, 1032, 0,
                           (struct sockaddr*)&clientAddr, sizeof(clientAddr))) < 0) {
                        fprintf(stderr, "Error sending raw data to client %d\n", result);
                        exit(EXIT_FAILURE);
                    }

                    raw_sequence++;
                    rawframeoffset=8;
                }
            }
        }
        
    }
}
