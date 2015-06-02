/*
 * Utils.h
 *
 *  Created on: 12-05-2015
 *      Author: adam
 */

#ifndef UTILS_H_
#define UTILS_H_

float* ReadFile(const char* name, int* count);
void testIO();
int initCuda(int argc, char ** argv);



double mclock();

#define NEXT_ARRAY(array, size) &(array[size])

#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaThreadSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

#define CHECK_SINGLE_ERROR(){\
		cudaError_t err = cudaGetLastError();                             \
		    if (cudaSuccess != err) {                                         \
		        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
		                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
		        exit(EXIT_FAILURE);                                           \
		    }}

struct Timer {
        clock_t lastTick;
        double duration;
};

Timer createTimer();

float tickTimer(Timer* timer);

#endif /* UTILS_H_ */
