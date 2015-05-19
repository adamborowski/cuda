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
#include "device_aggregator.cuh"
#include "common_utils.cuh"
double mclock() {
	struct timeval tp;
	double sec, usec;
	gettimeofday(&tp, NULL);
	sec = double(tp.tv_sec);
	usec = double(tp.tv_usec) / 1E6;
	return sec + usec;
}




__global__ void agg_10s_1m(int numSamples, const float* samples, int cacheSize, float * d_aggr_min, float * d_aggr_max, float * d_aggr_avg) {
	device_aggregate(0, 0, 0, 0, 0, 0, 0, 0);
	const int globalID = blockDim.x * blockIdx.x + threadIdx.x;
	const int localID = threadIdx.x;
	const int threadsPerBlock = blockDim.x;
#ifdef DEBUG
	if (globalID == 0) {
		int numThreads = blockDim.x * gridDim.x;
		printf("real num threads: %d\n", numThreads);
		printf("kernel address device min: %p, max: %p, avg: %p\n", d_aggr_min, d_aggr_max, d_aggr_avg);
	}
#endif
	int numChunks;
	int chunkSize, thisChunkSize;
	extern __shared__ float shared[];
	float* s_min = shared;
	float* s_max = &(s_min[threadsPerBlock]);
	float* s_avg = &(s_max[threadsPerBlock]);

	/**
	 * AGG_1s to AGG_10s
	 * - this aggregation reads from global memory
	 * - then calculates min, max, avg
	 * - and stores in shared memory as 10s aggregations
	 */
	chunkSize = AGG_SEC_10 / AGG_SEC_1;
	thisChunkSize = getChunkSize(numSamples, chunkSize, globalID);
	numChunks = divceil(threadsPerBlock, chunkSize);

	if (thisChunkSize > 0) { //jeśli wątek ma co robić
		float aMin = INFINITY, aMax = -1 * INFINITY, aAvg = 0;
		int globalIndex = globalID * chunkSize;
		float tmp;
		for (int i = 0; i < thisChunkSize; i++) { //we cannot exceed own chunk and the non-aligned array
			tmp = samples[globalIndex + i]; //get global element
			if (tmp < aMin)
				aMin = tmp;
			if (tmp > aMax)
				aMax = tmp;
			aAvg += tmp;
		}
		aAvg /= thisChunkSize;
		//write to shared memory
		s_min[localID] = aMin;
		s_max[localID] = aMax;
		s_avg[localID] = aAvg;
#ifdef DEBUG
		//shared memory write tests
		if (s_min[localID] != aMin)
			printf("failed to write to shared s_min at lid %d \t %f != %f\n", localID, s_min[localID], aMin);
		if (s_max[localID] != aMax)
			printf("failed to write to shared s_max at lid %d \t %f != %f\n", localID, s_max[localID], aMax);
		if (s_avg[localID] != aAvg)
			printf("failed to write to shared s_avg at lid %d \t %f != %f\n", localID, s_avg[localID], aAvg);
#endif
		//also write to global memory as a result of AGG_10s
		int offset10s = getAggOffset(numSamples, AGG_SEC_10); //we got offset to write 10s aggreations
		int globalPtr = offset10s + globalID;
		d_aggr_min[globalPtr] = aMin;
		d_aggr_max[globalPtr] = aMax;
		d_aggr_avg[globalPtr] = aAvg;
#ifdef DEBUG
		//global memory write tests
		if (d_aggr_min[globalPtr] != aMin)
			printf("failed to write to global d_aggr_min at gidx %d \t %f != %f\n", globalPtr, d_aggr_min[globalPtr], aMin);
		if (d_aggr_max[globalPtr] != aMax)
			printf("failed to write to global d_aggr_max at gidx %d \t %f != %f\n", globalPtr, d_aggr_max[globalPtr], aMax);
		if (d_aggr_avg[globalPtr] != aAvg)
			printf("failed to write to global d_aggr_acg at gidx %d \t %f != %f\n", globalIndex, d_aggr_avg[globalIndex], aAvg);
#endif
	}

	if (globalID < 10) {

		printf("min(0) = %f, max(0) = %f, avg(0) = %f\n", d_aggr_min[globalID], d_aggr_max[globalID], d_aggr_avg[globalID]);
	}
	__syncthreads();
	/**
	 * AGG_SEC_10 to AGG_MIN from shared to shared, then copy to global
	 */
	chunkSize = AGG_MIN / AGG_SEC_10;
	int lastNumChunks = numChunks; //z ilu poprzednich mini-chunków mamy generować nowy mega-chunk
	thisChunkSize = getChunkSize(lastNumChunks, chunkSize, localID);
	numChunks = divceil(threadsPerBlock, chunkSize);
	int tmpMin, tmpMax, tmpAvg;

	float aMin = INFINITY, aMax = -1 * INFINITY, aAvg = 0;
	//pobierz wskaźnik na chunk w pamięci shared po poprzednim zadaniu
	int sharedIndex = localID * chunkSize;

	for (int i = 0; i < thisChunkSize; i++) {
		//pobierz wartości kolejnych elementów chunków ale rozróżnij tablice min max avg
		tmpMin = s_min[sharedIndex + i];
		tmpMax = s_max[sharedIndex + i];
		tmpAvg = s_avg[sharedIndex + i];
		if (tmpMin < aMin)
			aMin = tmpMin;
		if (tmpMax > aMax)
			aMax = tmpMax;
		aAvg += tmpAvg;
		if (globalID == 0) {
			printf("chunk size: %d\n", chunkSize);
			printf("shared Index: %d\n", sharedIndex);
			printf("avg %f \n", aAvg);
		}
	}

	aAvg /= thisChunkSize;

	__syncthreads(); //upewnij się, że każdy wątek już odczytał pamięć shared aby można było zacząć do niej pisać
	if (thisChunkSize > 0) { //jeśli wątek ma co robić
		//zapisuj do shared
		s_min[localID] = aMin;
		s_max[localID] = aMax;
		s_avg[localID] = aAvg;
		//zapisuj do global jako wynik
		int offset1m = getAggOffset(numSamples, AGG_MIN); //we got offset to write 10s aggreations
		int globalPtr = offset1m + globalID;

		d_aggr_min[globalPtr] = aMin;
		d_aggr_max[globalPtr] = aMax;
		d_aggr_avg[globalPtr] = aAvg;
		if (globalID < 10) {
			printf("min(0) = %f, max(0) = %f, avg(0) = %f\n", d_aggr_min[globalPtr], d_aggr_max[globalPtr], d_aggr_avg[globalPtr]);
		}
	}

	__syncthreads();
	if (globalID == 0) {
		printf("pomyslnie skonczylo sie");
	}
}

void process(const char* name, int argc, char **argv) {
//nowe deklaracje
	int numSamples, aggHeapSize;
	float *h_samples, *h_aggr_min, *h_aggr_max, *h_aggr_avg;
	float *d_samples, *d_aggr_min, *d_aggr_max, *d_aggr_avg;
// This will pick the best possible CUDA capable device
	initCuda(argc, argv);
//allocate memory on cpu
	h_samples = ReadFile(name, &numSamples);
	aggHeapSize = getAggOffset(numSamples, AGG_ALL) * sizeof(float);
#ifdef DEBUG
	printf("numSamples = %d\n", numSamples);
	printf("heapSize = %d\n", aggHeapSize);

#endif
//allocate memory on gpu
	checkCudaErrors(cudaMalloc((void ** ) &d_samples, numSamples * sizeof(float)));	//todo zrobić cudaMalloc dla poszczególnych agregacji
	checkCudaErrors(cudaMalloc((void ** ) &d_aggr_min, aggHeapSize));
	checkCudaErrors(cudaMalloc((void ** ) &d_aggr_max, aggHeapSize));
	checkCudaErrors(cudaMalloc((void ** ) &d_aggr_avg, aggHeapSize));

//transfer samples from cpu to gpu
	cudaMemcpy(d_samples, h_samples, numSamples * sizeof(float), cudaMemcpyHostToDevice);

	int threadsPerBlock = 512;
//tworzymy tyle wątków ile potrzeba do policzenia najmniejszej agregacji
	int blocksPerGrid = divceil(getAggCount(numSamples, AGG_SEC_10), threadsPerBlock);
	int cacheSize = threadsPerBlock * sizeof(float) * NUM_AGGREGATORS;	//every thread calculates AGG_SEC_10{min, max,avg}

#ifdef DEBUG
	printf("host address device min: %p, max: %p, avg: %p\n", d_aggr_min, d_aggr_max, d_aggr_avg);
	printf("threadsPerBlock = %d, blocksPerGrid = %d, totalThreads = %d, sharedSize = %d\n", threadsPerBlock, blocksPerGrid, threadsPerBlock * blocksPerGrid, cacheSize);
#endif
	agg_10s_1m<<<blocksPerGrid, threadsPerBlock, cacheSize>>>(numSamples, d_samples, cacheSize, d_aggr_min, d_aggr_max, d_aggr_avg);
	cudaFree(d_samples);
	cudaFree(d_aggr_min);
	cudaFree(d_aggr_max);
	cudaFree(d_aggr_avg);
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
