
//#define DFC_FFT_INPUT_SIZE 1048576
//#define DFC_FFT_OVERLAP_SIZE 262144
//#define DFC_FFT_OUTPUT_SIZE ((DFC_FFT_INPUT_SIZE / 2) + 1)

extern cufftComplex* frequencysamples;
extern cufftComplex* deviceFrequencysamples;

extern sem_t frequencyBufferFull;
extern sem_t frequencyBufferEmpty;

void initDfc(void);
void dfcTerminate(void);

