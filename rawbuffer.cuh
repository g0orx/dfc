#define RAW_BUFFER_SIZE 4096

extern short* rawBuffer;
extern short* deviceRawBuffer;

extern sem_t rawBufferEmpty;
extern sem_t rawBufferFull;


void initRawBuffer(void);
