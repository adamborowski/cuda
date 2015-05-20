#include "Utils.h"
#include "CudaProj.h"
#include "common_utils.cuh"
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
#include "kernels.cuh"

void process(const char* name, int argc, char **argv) {
//nowe deklaracje
	int numSamples, aggHeapCount, aggHeapSize;
	float *h_samples, *h_aggr_min, *h_aggr_max, *h_aggr_avg;
	float *d_samples, *d_aggr_min, *d_aggr_max, *d_aggr_avg;
// This will pick the best possible CUDA capable device
	initCuda(argc, argv);
//allocate memory on cpu
	h_samples = ReadFile(name, &numSamples);

	aggHeapCount = getAggOffset(numSamples, AGG_ALL);
	aggHeapSize = aggHeapCount * sizeof(float);
	h_aggr_min = (float*) malloc(aggHeapSize);
	h_aggr_max = (float*) malloc(aggHeapSize);
	h_aggr_avg = (float*) malloc(aggHeapSize);
	cleanArray(aggHeapCount, h_aggr_min);
	cleanArray(aggHeapCount, h_aggr_max);
	cleanArray(aggHeapCount, h_aggr_avg);
#ifdef DEBUG
	printf("numSamples = %d\n", numSamples);
	printf("heapCount = %d\n", aggHeapCount);
	printf("heapSize = %d\n", aggHeapSize);

#endif
//allocate memory on gpu
	checkCudaErrors(cudaMalloc((void ** ) &d_samples, numSamples * sizeof(float)));	//todo zrobić cudaMalloc dla poszczególnych agregacji
	checkCudaErrors(cudaMalloc((void ** ) &d_aggr_min, aggHeapSize));
	checkCudaErrors(cudaMalloc((void ** ) &d_aggr_max, aggHeapSize));
	checkCudaErrors(cudaMalloc((void ** ) &d_aggr_avg, aggHeapSize));
	//clean gpu arrays
	checkCudaErrors(cudaMemcpy(d_aggr_min, h_aggr_min, aggHeapSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_aggr_max, h_aggr_max, aggHeapSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_aggr_avg, h_aggr_avg, aggHeapSize, cudaMemcpyHostToDevice));

//transfer samples from cpu to gpu
	checkCudaErrors(cudaMemcpy(d_samples, h_samples, numSamples * sizeof(float), cudaMemcpyHostToDevice));

//	int threadsPerBlock = 512;
	int threadsPerBlock = AGG_TEST_18;	//TODO zoptymalizować ( AGG_TEST_18/AGG_TEST_3 )
//tworzymy tyle wątków ile potrzeba do policzenia najmniejszej agregacji
	int blocksPerGrid = divceil(getAggCount(numSamples, AGG_SAMPLE), threadsPerBlock);
	int cacheSize = threadsPerBlock * sizeof(float) * NUM_AGGREGATORS;	//every thread calculates AGG_SEC_10{min, max,avg}

#ifdef DEBUG
//	printf("host address device min: %p, max: %p, avg: %p\n", d_aggr_min, d_aggr_max, d_aggr_avg);
	printf("threadsPerBlock = %d, blocksPerGrid = %d, totalThreads = %d, sharedSize = %d\n", threadsPerBlock, blocksPerGrid, threadsPerBlock * blocksPerGrid, cacheSize);
#endif
	agg_kernel_1<<<blocksPerGrid, threadsPerBlock, cacheSize>>>(numSamples, d_samples, cacheSize, d_aggr_min, d_aggr_max, d_aggr_avg);
	//wywołanie kernela zbierającego dane z niezależnych bloków (zatem mamy tylko jeden blok)
	threadsPerBlock = blocksPerGrid;
	blocksPerGrid = 1;
	cacheSize = threadsPerBlock * sizeof(float) * NUM_AGGREGATORS;
	agg_kernel_2<<<blocksPerGrid, threadsPerBlock, cacheSize>>>(numSamples, cacheSize, d_aggr_min, d_aggr_max, d_aggr_avg);
	checkCudaErrors(cudaMemcpy(h_aggr_min, d_aggr_min, aggHeapSize, cudaMemcpyDeviceToHost));
//	printf("\nskopiowalem aggr min aggHeapCount: %d\n", aggHeapCount);
	checkCudaErrors(cudaMemcpy(h_aggr_max, d_aggr_max, aggHeapSize, cudaMemcpyDeviceToHost));
//	printf("\nskopiowalem aggr max\n");
	checkCudaErrors(cudaMemcpy(h_aggr_avg, d_aggr_avg, aggHeapSize, cudaMemcpyDeviceToHost));
//	printf("\nskopiowalem aggr avg\n");
	printHeap(numSamples, h_aggr_min);
	cudaFree(d_samples);
	cudaFree(d_aggr_min);
	cudaFree(d_aggr_max);
	cudaFree(d_aggr_avg);
	cudaDeviceReset();
	printf("\n\n------------------ END ------------------\n");
	free(h_samples);
	free(h_aggr_min);
	free(h_aggr_max);
	free(h_aggr_avg);
}
int main(int argc, char **argv) {
//	int size = 20;
//	;
//	printf("offset of 1: %d\n", getAggOffset(size, AGG_TEST_1));
//	printf("offset of 3: %d\n", getAggOffset(size, AGG_TEST_3));
//	printf("offset of 6: %d\n", getAggOffset(size, AGG_TEST_6));
//	printf("offset of 18: %d\n", getAggOffset(size, AGG_TEST_18));
//	printf("offset of 36: %d\n", getAggOffset(size, AGG_TEST_36));
//	printf("offset of 108: %d\n", getAggOffset(size, AGG_TEST_108));
//	printf("heap count: %d\n", getAggOffset(size, AGG_ALL));
#ifdef TEST
//	process("Test_data.txt", argc, argv);
	process("data/Osoba_cut.txt", argc, argv);
#else
	process("data/Osoba_concat.txt", argc, argv);
#endif
	CHECK_LAUNCH_ERROR()
				;
}
