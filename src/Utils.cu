#include "Utils.h"
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
	while (fgets(line, 1024, file)) {
		sscanf(line, "%*s %f", &buffer[_count++]);
	}
	float* goodArray = (float*) malloc(sizeof(float) * _count);
	memcpy(goodArray, buffer, _count * sizeof(float));
	*count = _count;
	fclose(file);
	free(buffer);
	//Do what ever with buffer

	return goodArray;
}
void testIO() {
	int count;
	float* buffer = ReadFile("data/Osoba_concat.txt", &count);
	printf("Lines loaded: %d\n", count);
}

int divceil(int a, int b) {
	return (a + b - 1) / b;
}

void initCuda(int argc, char ** argv) {
	cudaDeviceProp deviceProp;
	int devID = findCudaDevice(argc, (const char **) argv);

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

}


