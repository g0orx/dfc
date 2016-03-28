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

#include "common.cuh"
#include "inputbuffer.cuh"
#include "socket.cuh"
#include "time.cuh"

#define DATA_PROTO 0xefff

static int sockFd;

static pthread_t socketThread;

void* socketLoop(void* args);

void initSocket() {
    struct ifreq ifbuffer;
    int result;
    int sockopt;

    fprintf(stderr,"initSocket: %s\n", interface);

    if ((sockFd = socket(AF_PACKET, SOCK_DGRAM, htons(DATA_PROTO))) < 0 ) {
        fprintf(stderr,"initSocket: error creating socket %d\n", sockFd);
        exit(EXIT_FAILURE);
    }

    memset(&ifbuffer, 0x00, sizeof(ifbuffer));
    strncpy(ifbuffer.ifr_name, interface, IFNAMSIZ-1);
    ioctl(sockFd, SIOCGIFFLAGS, &ifbuffer);
    ifbuffer.ifr_flags |= IFF_PROMISC;
    ioctl(sockFd, SIOCSIFFLAGS, &ifbuffer);

    sockopt=1;
    if((result=setsockopt(sockFd, SOL_SOCKET, SO_REUSEADDR, &sockopt, sizeof sockopt)) < 0) {
        fprintf(stderr,"initSocket: error setsockopt(SO_REUSEADDR) %d\n", result);
        exit(EXIT_FAILURE);
    }
    if((result=setsockopt(sockFd, SOL_SOCKET, SO_BINDTODEVICE, interface, IFNAMSIZ-1)) < 0) {
        fprintf(stderr,"initSocket: error setsockopt(SO_BINDTODEVICE) %d\n", result);
        exit(EXIT_FAILURE);
    }

    result=pthread_create(&socketThread, NULL, socketLoop, NULL);
    if(result<0) {
        fprintf(stderr, "Error creating socket thread: %d\n", result);
        exit(EXIT_FAILURE);
    }
}

void* socketLoop(void* args) {
    int bytestoread=L_SIZE*sizeof(short);
    int count;
    int offset;
    int result;

#ifdef TIMING
    long long starttime;
    long long endtime;
    int reads;
#endif
fprintf(stderr,"socketLoop: running on cpu %d\n", sched_getcpu());
//    fprintf(stderr, "socketLoop: bytestoread=%d\n",bytestoread);
    while(1) {
        result=sem_wait(&inputBufferEmpty);
        if(result!=0) {
            fprintf(stderr,"socketLoop: sem_wait failed for inputBufferEmpty %d\n", result);
            exit(EXIT_FAILURE);
        }
#ifdef TIMING
        starttime=current_timestamp();
        reads=0;
#endif
        offset=0;
        while(offset!=bytestoread) {
            count=recv(sockFd, &(inputBuffer[offset/sizeof(short)]), BYTES_PER_FRAME,0);
//            fprintf(stderr,"read %d bytes to %d at %p\n",count,offset/sizeof(short), &(inputBuffer[offset/sizeof(short)]));
            if(count<=0) {
                if(errno != EINTR) {
                    fprintf(stderr, "socketLoop error reading: %s\n", strerror(errno));
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
            fprintf(stderr,"socketLoop: sem_post failed for inputBufferFull %d\n", result);
            exit(EXIT_FAILURE);
        }
    }
}

