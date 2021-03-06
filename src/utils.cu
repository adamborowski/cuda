#include "utils.h"
#include "CudaProj.h"
#include <stdlib.h>
#include <stdio.h>
#include <helper_functions.h>  // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>       // helper function CUDA error checking and intialization
float* ReadFile(const char* name, int* count) {
	FILE *file;
	float *buffer;
	unsigned long fileLen;

	//Open file
	file = fopen(name, "rb");
	if (!file) {
		fprintf(stderr, "Unable to open file %s", name);
		exit(1);
	}

	//Get file length
	fseek(file, 0, SEEK_END);
	fileLen = ftell(file);
	fseek(file, 0, SEEK_SET);

	//Allocate memory
	buffer = (float *) malloc(fileLen + 1);

	if (!buffer) {
		fprintf(stderr, "Memory error!");
		fclose(file);
		exit(2);
	}

	//Read file contents into buffer
	char line[1024];
	int _count = 0;

	float testMin = INFINITY;

	while (fgets(line, 1024, file)) {
		sscanf(line, "%*s %f", &buffer[_count]);
		//TODO remove below line
		buffer[_count] = _count;
		if (buffer[_count] < testMin) {
			testMin = buffer[_count];
		}
		_count++;
	}
	_count = (_count / AGG_TEST_108) * AGG_TEST_108; //FIXME wyrównanie
	printf("\n~~~~~~~~~~~~~~ TEST MIN: %f ~~~~~~~~~~~~~~\n", testMin);
	float* goodArray = (float*) malloc(sizeof(float) * _count);
	memcpy(goodArray, buffer, _count * sizeof(float));
	*count = _count;
	fclose(file);
	free(buffer);
	//Do what ever with buffer

	return goodArray;
}

float* getMockData(const int count) {
	float* buffer = (float *) malloc(count*sizeof(float));
	for (int i = 0; i < count; i++) {
		buffer[i] = i;
	}
	return buffer;
}

void testIO() {
	int count;
	float* buffer = ReadFile("data/Osoba_concat.txt", &count);
	printf("Lines loaded: %d\n", count);
}

int initCuda(int argc, char ** argv) {
	cudaDeviceProp deviceProp;

	int devID = 0;
	if (argc >= 6) {
		devID = atoi(argv[5]);
	}

	printf("\n====================\nworking on device ID: %d\n====================\n", devID);

	cudaSetDevice(devID);

	if (devID < 0) {
		printf("exiting...\n");
		exit(EXIT_SUCCESS);
	}

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

	// Statistics about the GPU device
	printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n", deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	int version = (deviceProp.major * 0x10 + deviceProp.minor);

	if (version < 0x11) {
		printf("%s: requires a minimum CUDA compute 1.1 capability\n", "Adam Borowski");
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}
	return devID;
}

double mclock() {
	struct timeval tp;
	double sec, usec;
	gettimeofday(&tp, NULL);
	sec = double(tp.tv_sec);
	usec = double(tp.tv_usec) / 1E6;
	return sec + usec;
}

Timer createTimer() {
	Timer timer;
	cudaEventCreate(&timer.startEvent);
	cudaEventCreate(&timer.stopEvent);
	cudaEventRecord(timer.startEvent, 0);
	timer.duration = 0;
	return timer;
}

float tickTimer(Timer* timer) {
	cudaEventRecord(timer->stopEvent, 0);
	cudaEventSynchronize(timer->stopEvent);
	cudaEventElapsedTime(&timer->duration, timer->startEvent, timer->stopEvent);
	return timer->duration;
}

