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
#include "file.cuh"
#include "time.cuh"

static int rawFd;
static pthread_t fileThread;

void* fileLoop(void* args);

void initFile() {

    fprintf(stderr, "init_file\n");
    rawFd=open("raw.bin", O_RDONLY);
    if(rawFd<0) {
        perror("Failed to open raw.bin");
        exit(EXIT_FAILURE);
    }

    int result=pthread_create(&fileThread, NULL, fileLoop, NULL);
    if(result<0) {
        fprintf(stderr, "Error creating file thread: %d\n", result);
        exit(EXIT_FAILURE);
    }
}

void* fileLoop(void* args) {
    int bytestoread=L_SIZE*sizeof(short);
    int count;
    int offset;
    int result;

#ifdef TIMING
    long long starttime;
    long long endtime;
    int reads;
#endif
    fprintf(stderr,"fileLoop: running on cpu %d\n", sched_getcpu());
    while(1) {
        result=sem_wait(&inputBufferEmpty);
        if(result!=0) {
            fprintf(stderr,"fileLoop: sem_wait failed for inputBufferEmpty %d\n", result);
            exit(EXIT_FAILURE);
        }
#ifdef TIMING
        starttime=current_timestamp();
        reads=0;
#endif
        offset=0;
        while(offset!=bytestoread) {
            count=read(rawFd, &inputBuffer[offset/sizeof(short)], bytestoread-offset);
            if(count<=0) {
                if(errno != EINTR) {
                    //fprintf(stderr, "fileLooperror error reading: %s\n", strerror(errno));
                    close(rawFd);
                    rawFd=open("raw.bin", O_RDONLY);
                    if(rawFd<0) {
                        perror("Failed to re-open raw.bin");
                        exit(EXIT_FAILURE);
                    }
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
            fprintf(stderr,"fileLoop: sem_post failed for inputBufferFull %d\n", result);
            exit(EXIT_FAILURE);
        }
    }
}

