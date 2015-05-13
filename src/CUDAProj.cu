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

__global__ void dot1(int N, const float *a, const float *b, float *partialSum) {

	//tutaj odpala się kilka wątków w jednym bloku<<
	extern __shared__ float c_shared[];
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int localI = threadIdx.x;
	int globalI = blockIdx.x;
	//wymnazanie
	if (i < N) {
		c_shared[localI] = a[i] * b[i];
//		printf("c_shared[localI %d]=%f * %f\n", i, a[i], b[i]);
	}

	int leftPtr, rightPtr;
	int blockN = blockDim.x;
	//jeśli jest to ostatni block, to ma on mniej lub tyle samo jeśli jest całkowita wielokrotność
	if (blockIdx.x == gridDim.x - 1)
		blockN = N % blockN;
	if (blockN == 0)
		blockN = N;
//	printf("block %d blockN %d", blockIdx.x, blockN);
	//poniżej nie N tylko ilość thread in block

	for (int skip = 1; skip < blockN; skip <<= 1) {
		__syncthreads();

		if (i < N) { //suma czesciowa
			leftPtr = localI * skip * 2;
			rightPtr = leftPtr + skip;
			if (rightPtr < blockN) {
//				printf(">>DOT1 %d.%d skip: %d, %d+%d, %.0f+%.0f\n", blockIdx.x, threadIdx.x, skip, leftPtr, rightPtr, c_shared[leftPtr], c_shared[rightPtr]);
				c_shared[leftPtr] += c_shared[rightPtr];
			}
		}

	}

//    __syncthreads();//nie trzeba syncrhonizować, wątek 0 ma wszystko co potrzeba
	if (localI == 0 && i < N) {
		partialSum[globalI] = c_shared[0];
//		printf("part %d = %f\n", globalI, partialSum[globalI]);
	}
//    printf("****************** DOT1 finish i=%d, ", i);
}

/**
 * Poniższa funkcja robi redukcję a wynik zostaje w pierwszym elemencie
 * musi być wykonana w jednym bloku, jeśli wątków będzie za mało, trzeba w pętli wykonać jakieś machlojstwo
 */__global__ void dot2(int vectorSize, float *partialSum, int numBlocks) {
//    printf(">>DOT2\n");
	extern __shared__ float c_shared[];
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	//cache:
//    printf("to shared i=%d: %f", i, partialSum[i]);
	if (i < numBlocks) {
//		printf("to shared i=%d: %f\n", i, partialSum[i]);
		c_shared[i] = partialSum[i];
	}

	int numParts = numBlocks;
	if (numParts > vectorSize)
		numParts = vectorSize;
	int localI = i;
	int skip;
	int leftPtr, rightPtr;

	for (skip = 1; skip < numParts; skip <<= 1) {
		__syncthreads();
		if (i < numParts) { //suma czesciowa
			leftPtr = localI * skip * 2;
			rightPtr = leftPtr + skip;
			if (rightPtr < numParts) {
//				printf(">>DOT1 %d.%d skip: %d, %d+%d\n", blockIdx.x, threadIdx.x, skip, leftPtr, rightPtr);
				c_shared[leftPtr] += c_shared[rightPtr];
			}
		}
	}

//    __syncthreads();//wątek zerowy zapisuje już do c_shared[0] dlatego nie trzeba synchronizować
//    printf("i: %d %f\n", i, c_shared[0]);
	if (i == 0) {
		partialSum[0] = c_shared[0];
//		printf("DOT2: %f\n", partialSum[0]);
	}

}

__global__ void agg_10s_1m(int numSamples, float* samples, int cacheSize, float * d_aggr_10s_min, float * d_aggr_10s_max, float * d_aggr_10s_avg) {
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
		if (s_min[globalID] != aMin)
			printf("failed to write to shared s_min at tid %d", globalID);
		if (s_max[globalID] != aMax)
			printf("failed to write to shared s_max at tid %d", globalID);
		if (s_avg[globalID] != aAvg)
			printf("failed to write to shared s_avg at tid %d", globalID);
#endif
		//also write to global memory as a result of AGG_10s
		d_aggr_10s_min[globalID] = aMin;
		d_aggr_10s_max[globalID] = aMax;
		d_aggr_10s_avg[globalID] = aAvg;
	}
	if (globalID == 0) {

		printf("min(0) = %f, max(0) = %f, avg(0) = %f\n", s_min[0], s_max[0], s_avg[0]);
	}

}

void process(const char* name, int argc, char **argv) {
	//nowe deklaracje
	int numSamples;
	float *h_samples, *h_aggr_10s_min, *h_aggr_10s_max, *h_aggr_10s_avg;
	float *d_samples, *d_aggr_10s_min, *d_aggr_10s_max, *d_aggr_10s_avg;
	// This will pick the best possible CUDA capable device
	initCuda(argc, argv);
	//allocate memory on cpu
	h_samples = ReadFile(name, &numSamples);
#ifdef DEBUG
	printf("numSamples = %d\n", numSamples);
#endif
	//allocate memory on gpu
	checkCudaErrors(cudaMalloc((void ** ) &d_samples, numSamples * sizeof(float)));	//todo zrobić cudaMalloc dla poszczególnych agregacji
	checkCudaErrors(cudaMalloc((void ** ) &d_aggr_10s_min, divceil(numSamples * sizeof(float),AGG_SEC_10)));	//todo zrobić cudaMalloc dla poszczególnych agregacji
	checkCudaErrors(cudaMalloc((void ** ) &d_aggr_10s_max, divceil(numSamples * sizeof(float),AGG_SEC_10)));	//todo zrobić cudaMalloc dla poszczególnych agregacji
	checkCudaErrors(cudaMalloc((void ** ) &d_aggr_10s_avg, divceil(numSamples * sizeof(float),AGG_SEC_10)));	//todo zrobić cudaMalloc dla poszczególnych agregacji
	//transfer samples from cpu to gpu
	cudaMemcpy(d_samples, h_samples, numSamples * sizeof(float), cudaMemcpyHostToDevice);

	int threadsPerBlock = 1024;
	//tworzymy tyle wątków ile potrzeba do policzenia najmniejszej agregacji
	int blocksPerGrid = divceil(divceil(numSamples, AGG_SEC_10), threadsPerBlock);
	int cacheSize = NUM_AGGREGATORS * sizeof(float) * divceil(threadsPerBlock, AGG_SEC_10);

#ifdef DEBUG

	printf("threadsPerBlock = %d, blocksPerGrid = %d, totalThreads = %d, sharedSize = %d\n", threadsPerBlock, blocksPerGrid, threadsPerBlock * blocksPerGrid, cacheSize);
#endif

	agg_10s_1m<<<blocksPerGrid, threadsPerBlock, cacheSize>>>(numSamples, d_samples, cacheSize, d_aggr_10s_min, d_aggr_10s_max, d_aggr_10s_avg);
	cudaFree(d_samples);
	cudaDeviceReset();
	printf("\n\n------------------ END ------------------\n");

}
int main(int argc, char **argv) {
	process("data/Osoba_concat.txt", argc, argv);
}
