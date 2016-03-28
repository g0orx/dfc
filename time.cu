#include <sys/time.h>

long long current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL);
    long long milliseconds = te.tv_sec*1000LL + te.tv_usec/1000;
    return milliseconds;
}
