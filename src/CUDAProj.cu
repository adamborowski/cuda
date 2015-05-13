#define DEBUG
#include "Utils.h"
#include "CudaProj.h"
// includes, system

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
/* Using updated (v2) interfaces to cublas and cusparse */
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>       // helper function CUDA error checking and intialization
double mclock() {
	struct timeval tp;
	double sec, usec;
	gettimeofday(&tp, NULL);
	sec = double(tp.tv_sec);
	usec = double(tp.tv_usec) / 1E6;
	return sec + usec;
}
__host__ __device__ int getAggCount(const int numSamples, const int aggType) {
	return divceil(numSamples, aggType);
}
/**
 * Offset of tree heap. The smallest aggregation is at the begining.
 * 1s 1s 1s 1s 1s 1s 1s 1s 1s 1s 1s 1s 1s 1s 1s 1s...
 * 10s 10s 10s 10s 10s...
 * 1m 1m 1m 1m 1m...
 * 10m 10m 10m...
 * 30m 30m...
 * 1h 1h...
 * 24h...
 */
__host__ __device__ int getAggOffset(const int numSamples, const int aggType) {
	int offset = 0;
	switch (aggType) { //in this switch if any case will occur, we continue to each next case (don't break)
	case AGG_ALL:
		offset += getAggCount(numSamples, AGG_YEAR);
		/* no break */
	case AGG_YEAR:
		offset += getAggCount(numSamples, AGG_HOUR_24);
		/* no break */
	case AGG_HOUR_24:
		offset += getAggCount(numSamples, AGG_HOUR);
		/* no break */
	case AGG_HOUR:
		offset += getAggCount(numSamples, AGG_MIN_30);
		/* no break */
	case AGG_MIN_30:
		offset += getAggCount(numSamples, AGG_MIN_10);
		/* no break */
	case AGG_MIN_10:
		offset += getAggCount(numSamples, AGG_MIN);
		/* no break */
	case AGG_MIN:
		offset += getAggCount(numSamples, AGG_SEC_10);
		/* no break */
//	case AGG_SEC_10:
		//we don't put 1s so we shouldn't add offset for it
		//offset += divceil(numSamples, AGG_SEC_1);
//		;
		/* no break */
//	case AGG_SEC_1:
		//we have nothing before to add offset
//		;
	}
	return offset;
}

__global__ void agg_10s_1m(int numSamples, float* samples, int cacheSize, float * d_aggr_min, float * d_aggr_max, float * d_aggr_avg) {
	int globalID = blockDim.x * blockIdx.x + threadIdx.x;
	extern __shared__ float shared[];
	float* s_min = shared;
	int numChunks = cacheSize / sizeof(float) / NUM_AGGREGATORS;
	float* s_max = &(shared[numChunks]);
	float* s_avg = &(s_max[numChunks]);
#ifdef DEBUG
	if (globalID == 0) {
		int numThreads = blockDim.x * gridDim.x;
		printf("real num threads: %d\n", numThreads);
	}
#endif

	/**
	 * AGG_1s to AGG_10s
	 * - this aggregation reads from global memory
	 * - then calculates min, max, avg
	 * - and stores in shared memory as 10s aggregations
	 */
	const int chunkSize = AGG_SEC_10;
	if (globalID < numSamples / chunkSize) { //jeśli wątek ma co robić
		float aMin = INFINITY, aMax = -1 * INFINITY, aAvg = 0;

		int globalIndex = globalID * chunkSize;
		float tmp;
		for (int i = 0; i < chunkSize && i < numSamples; i++) { //we cannot exceed own chunk and the non-aligned array
			tmp = samples[globalIndex + i]; //get global element
			if (tmp < aMin)
				aMin = tmp;
			if (tmp > aMax)
				aMax = tmp;
			aAvg += tmp;
		}
		aAvg /= chunkSize;
		//write to shared memory
		s_min[globalID] = aMin; //TODO block id zamiast globalId???
		s_max[globalID] = aMax;
		s_avg[globalID] = aAvg;
#ifdef DEBUG
		//shared memory write tests
		printf("jestem");
		if (s_min[globalID] != aMin)
			printf("failed to write to shared s_min at tid %d", globalID);
		if (s_max[globalID] != aMax)
			printf("failed to write to shared s_max at tid %d", globalID);
		if (s_avg[globalID] != aAvg)
			printf("failed to write to shared s_avg at tid %d", globalID);
#endif
		//also write to global memory as a result of AGG_10s
		int offset10s = getAggOffset(numSamples, AGG_SEC_10);//we got offset to write 10s aggreations
		d_aggr_min[offset10s + globalID] = aMin;
		d_aggr_max[offset10s + globalID] = aMax;
		d_aggr_avg[offset10s + globalID] = aAvg;

	}
	if (globalID == 0) {

		printf("min(0) = %f, max(0) = %f, avg(0) = %f\n", s_min[0], s_max[0], s_avg[0]);
	}

}

void process(const char* name, int argc, char **argv) {
	//nowe deklaracje
	int numSamples;
	float *h_samples, *h_aggr_min, *h_aggr_max, *h_aggr_avg;
	float *d_samples, *d_aggr_min, *d_aggr_max, *d_aggr_avg;
	// This will pick the best possible CUDA capable device
	initCuda(argc, argv);
	//allocate memory on cpu
	h_samples = ReadFile(name, &numSamples);
#ifdef DEBUG
	printf("numSamples = %d\n", numSamples);
#endif
	//allocate memory on gpu
	checkCudaErrors(cudaMalloc((void ** ) &d_samples, numSamples * sizeof(float)));	//todo zrobić cudaMalloc dla poszczególnych agregacji
	checkCudaErrors(cudaMalloc((void ** ) &d_aggr_min, getAggOffset(numSamples,AGG_ALL)));
	checkCudaErrors(cudaMalloc((void ** ) &d_aggr_max, getAggOffset(numSamples,AGG_ALL)));
	checkCudaErrors(cudaMalloc((void ** ) &d_aggr_avg, getAggOffset(numSamples,AGG_ALL)));
	//transfer samples from cpu to gpu
	cudaMemcpy(d_samples, h_samples, numSamples * sizeof(float), cudaMemcpyHostToDevice);

	int threadsPerBlock = 1024;
	//tworzymy tyle wątków ile potrzeba do policzenia najmniejszej agregacji
	int blocksPerGrid = divceil(divceil(numSamples, AGG_SEC_10), threadsPerBlock);
	int cacheSize = NUM_AGGREGATORS * sizeof(float) * divceil(threadsPerBlock, AGG_SEC_10);

#ifdef DEBUG

	printf("threadsPerBlock = %d, blocksPerGrid = %d, totalThreads = %d, sharedSize = %d\n", threadsPerBlock, blocksPerGrid, threadsPerBlock * blocksPerGrid, cacheSize);
#endif

	agg_10s_1m<<<blocksPerGrid, threadsPerBlock, cacheSize>>>(numSamples, d_samples, cacheSize, d_aggr_min, d_aggr_max, d_aggr_avg);
	cudaFree(d_samples);
	cudaDeviceReset();
	printf("\n\n------------------ END ------------------\n");

}
int main(int argc, char **argv) {
//	int size = 30 * 10 * 1000;
//	printf("offset of 10s: %d\n", getAggOffset(size, AGG_SEC_10));
//	printf("offset of 1m: %d\n", getAggOffset(size, AGG_MIN));
//	printf("offset of 10m: %d\n", getAggOffset(size, AGG_MIN_10));
//	printf("offset of 30m: %d\n", getAggOffset(size, AGG_MIN_30));
//	printf("offset of 1h: %d\n", getAggOffset(size, AGG_HOUR));
//	printf("offset of 24h: %d\n", getAggOffset(size, AGG_HOUR_24));
//	printf("offset of year: %d\n", getAggOffset(size, AGG_YEAR));
//	printf("offset of all: %d\n", getAggOffset(size, AGG_ALL));

	process("data/Osoba_concat.txt", argc, argv);
}
