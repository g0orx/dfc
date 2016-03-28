#include <cufft.h>

//#define TIMING

#define SOCKET_SAMPLE_RATE 61440000
#define PCIE_SAMPLE_RATE 73529412
#define DIVISOR (32767.0F)


#define P_SIZE                  262145                          // FIR Length
#define V_SIZE                  4                               // Overlap factor  V = N/(P-1)
#define DFT_BLOCK_SIZE          ((P_SIZE - 1) * V_SIZE)         // N real samples
#define L_SIZE                  (DFT_BLOCK_SIZE - P_SIZE + 1)   // Number of new input samples consumed per data block

#define COMPLEX_SIGNAL_SIZE     (DFT_BLOCK_SIZE / 2)            // N/2

#define NFACTOR (double)(COMPLEX_SIGNAL_SIZE) / ((double)V_SIZE * (samplingrate / 2))

#define D_SIZE_48K              128
#define D_SIZE_96K              64
#define D_SIZE_192K             32
#define D_SIZE_384K             16

#define BYTES_PER_FRAME         8192
#define SAMPLES_PER_FRAME       (BYTES_PER_FRAME / sizeof(short))
#define FRAMES_PER_BUFFER       (L_SIZE / SAMPLES_PER_FRAME)

#define IFFT_DECIMATE_MAX       (samplingrate/D_SIZE_384K/2/48000)
#define IFFT_DECIMATE_MIN       (samplingrate/D_SIZE_48K/2/48000)

#define RX_TD_SIZE              ((COMPLEX_SIGNAL_SIZE/d_size - (P_SIZE-1)/2/d_size)/ifft_decimate_factor + 1)
#define DEVICE_RX_TD_SIZE       ((COMPLEX_SIGNAL_SIZE/d_size - (P_SIZE-1)/2/d_size)/decimate + 1)
#define RX_TD_MAXSIZE           ((COMPLEX_SIGNAL_SIZE/D_SIZE_384K - (P_SIZE-1)/2/D_SIZE_384K) / IFFT_DECIMATE_MIN + 1)

#define SOURCE_PCIE 0
#define SOURCE_SOCKET 1
#define SOURCE_FILE 2

extern int source;
extern int capture;
extern int samplingrate;
extern float hzperbin;
extern char interface[];

// Complex multiplication
__device__ inline cufftComplex ComplexMul(const cufftComplex a, const cufftComplex b)
{
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;

    return (c);
}

// Complex scale
__device__ inline cufftComplex ComplexScale(const cufftComplex a, const float s)
{
    cufftComplex c;
    c.x = s * a.x;
    c.y = s * a.y;

    return (c);
}

__device__ inline cufftComplex ComplexAdd(const cufftComplex a, const cufftComplex b)
{
    cufftComplex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;

    return (c);
}

