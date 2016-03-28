extern bool audio;
extern int audio_buffer_size;

void audio_init();
//void audio_write(char *buffer, int length);
void audio_write(unsigned char *buffer);
