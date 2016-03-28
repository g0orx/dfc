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
