//#define INPUT_BUFFER_SIZE 786432
//#define INPUT_BUFFER_SIZE 1048576

extern short* inputBuffer;
extern short* deviceInputBuffer;

extern sem_t inputBufferEmpty;
extern sem_t inputBufferFull;


void initInputBuffer(void);
