#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>

#include <pulse/simple.h>
#include <pulse/error.h>

#include "audio.cuh"

bool audio = false;
int audio_buffer_size = 2016; // samples (both left and right)

static pa_simple *stream;

static sem_t audioBufferFull;
static sem_t audioBufferEmpty;
static pthread_t audioThreadId;

// each buffer contains 63 samples of left and right audio at 16 bits
#define AUDIO_SAMPLES 63
#define AUDIO_SAMPLE_SIZE 2
#define AUDIO_CHANNELS 2
#define AUDIO_BUFFERS 10
#define AUDIO_BUFFER_SIZE (AUDIO_SAMPLE_SIZE*AUDIO_CHANNELS*audio_buffer_size)

static unsigned char *audio_buffer;
static int audio_offset=0;

void* audioThread(void* arg);

void audio_init() {

    static const pa_sample_spec spec= {
        .format = PA_SAMPLE_S16RE,
        .rate =  48000,
        .channels = 2
    };

    int error;

fprintf(stderr,"audio_init audio_buffer_size=%d\n",audio_buffer_size);

    audio_buffer=(unsigned char *)malloc(AUDIO_BUFFER_SIZE);

    if (!(stream = pa_simple_new(NULL, "nghermes", PA_STREAM_PLAYBACK, NULL, "playback", &spec, NULL, NULL, &error))) {
        fprintf(stderr, __FILE__": pa_simple_new() failed: %s\n", pa_strerror(error));
        exit(1);
    }

    int res=sem_init(&audioBufferFull, 0, 0);
    if(res!=0) {
        fprintf(stderr,"audio_init: sem_init failed for audioBufferFull%d\n", res);
        exit(EXIT_FAILURE);
    }

    res=sem_init(&audioBufferEmpty, 0, 0);
    if(res!=0) {
        fprintf(stderr,"audio_init: sem_init failed for audioBufferEmpty%d\n", res);
        exit(EXIT_FAILURE);
    }

    res=pthread_create(&audioThreadId, NULL, audioThread, NULL);
    if(res<0) {
        fprintf(stderr, "Error creating DFC thread: %d\n", res);
        exit(EXIT_FAILURE);
    }

fprintf(stderr,"... audio_init\n");
}


void audio_write(unsigned char* buffer) {
    int i;
    int error;

    for(i=0;i<63;i++) {
        int source_index=8+(i*8);
        audio_buffer[audio_offset++]=buffer[source_index];
        audio_buffer[audio_offset++]=buffer[source_index+1];
        audio_buffer[audio_offset++]=buffer[source_index+2];
        audio_buffer[audio_offset++]=buffer[source_index+3];

        if(audio_offset==AUDIO_BUFFER_SIZE) {
            if (pa_simple_write(stream, audio_buffer, (size_t)AUDIO_BUFFER_SIZE, &error) < 0) {
                fprintf(stderr, __FILE__": pa_simple_write() failed: %s\n", pa_strerror(error));
                exit(1);
            }
            audio_offset=0;
        }
    }

}

void* audioThread(void* arg) {
    int error;
    fprintf(stderr,"audioThread running on cpu%d\n", sched_getcpu());

    while(1) {

/*
        error=sem_post(&audioBufferEmpty);
        if(error!=0) {
            fprintf(stderr, "audioThread: sem_post failed for audioBufferEmpty: %d\n", error);
            exit(EXIT_FAILURE);
        }
*/
        error=sem_wait(&audioBufferFull);
        if(error!=0) {
            fprintf(stderr, "audioThread: sem_wait failed for audioBufferFull: %d\n", error);
            exit(EXIT_FAILURE);
        }

        if (pa_simple_write(stream, audio_buffer, (size_t)AUDIO_BUFFER_SIZE, &error) < 0) {
            fprintf(stderr, __FILE__": pa_simple_write() failed: %s\n", pa_strerror(error));
            exit(1);
        }
    }
}
