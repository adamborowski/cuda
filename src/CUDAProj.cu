#include "utils.h"
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
#include "device_manager.cuh"

void process(const char* name, int argc, char **argv) {
	//nowe deklaracje
	int numSamples, aggHeapCount, aggHeapSize;
	float *h_samples, *h_aggr_min, *h_aggr_max, *h_aggr_avg;
	float *d_samples, *d_aggr_min, *d_aggr_max, *d_aggr_avg;
	// This will pick the best possible CUDA capable device
	int deviceId = initCuda(argc, argv);
	//allocate memory on cpu
	h_samples = ReadFile(name, &numSamples);

	aggHeapCount = getAggOffset(numSamples, AGG_ALL);
	aggHeapSize = aggHeapCount * sizeof(float);
	h_aggr_min = (float*) malloc(aggHeapSize);
	h_aggr_max = (float*) malloc(aggHeapSize);
	h_aggr_avg = (float*) malloc(aggHeapSize);
#ifdef DEBUG
	cleanArray(aggHeapCount, h_aggr_min);
	cleanArray(aggHeapCount, h_aggr_max);
	cleanArray(aggHeapCount, h_aggr_avg);
	printf("{HOST} numSamples = %d\n", numSamples);
	printf("{HOST} heapCount = %d\n", aggHeapCount);
	printf("{HOST} heapSize = %d\n", aggHeapSize);
#endif
//allocate memory on gpu
	checkCudaErrors(cudaMalloc((void ** ) &d_samples, numSamples * sizeof(float)));	//todo zrobić cudaMalloc dla poszczególnych agregacji
	checkCudaErrors(cudaMalloc((void ** ) &d_aggr_min, aggHeapSize));
	checkCudaErrors(cudaMalloc((void ** ) &d_aggr_max, aggHeapSize));
	checkCudaErrors(cudaMalloc((void ** ) &d_aggr_avg, aggHeapSize));
#ifdef DEBUG
	//clean gpu arrays
	checkCudaErrors(cudaMemcpy(d_aggr_min, h_aggr_min, aggHeapSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_aggr_max, h_aggr_max, aggHeapSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_aggr_avg, h_aggr_avg, aggHeapSize, cudaMemcpyHostToDevice));
#endif
	//transfer samples from cpu to gpu
	checkCudaErrors(cudaMemcpy(d_samples, h_samples, numSamples * sizeof(float), cudaMemcpyHostToDevice));
	//tworzymy tyle wątków ile potrzeba do policzenia najmniejszej agregacji
	int threadsPerBlock = SETTINGS_GROUP_A_SIZE + SETTINGS_GROUP_B_SIZE + SETTINGS_GROUP_C_SIZE;
	int blocksPerGrid = SETTINGS_NUM_BLOCKS;

	//calculate cacheSize
	BlockState bs;
	bs.partSize = AGG_TEST_108;
	bs.num_B_threads = SETTINGS_GROUP_B_SIZE;
	int cacheSize = initializeSharedMemory(&bs);

	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, deviceId);
	if (cacheSize > devProp.sharedMemPerBlock) {
		fprintf(stderr, "Requred cache (%dB) cannot be allocated. Only %dB allowed. Try decrease SETTINGS_GROUP_B_SIZE.", cacheSize, (int) devProp.sharedMemPerBlock);
		exit(-1);
	}

#ifdef DEBUG
//	printf("host address device min: %p, max: %p, avg: %p\n", d_aggr_min, d_aggr_max, d_aggr_avg);
	printf("{HOST} threadsPerBlock = %d, blocksPerGrid = %d, totalThreads = %d, cacheSize = %d\n", threadsPerBlock, blocksPerGrid, threadsPerBlock * blocksPerGrid, cacheSize);
#endif

	AggrPointers output;
	output.min = d_aggr_min;
	output.max = d_aggr_max;
	output.avg = d_aggr_avg;

	kernel_manager<<<blocksPerGrid, threadsPerBlock, cacheSize>>>(numSamples, d_samples, output);	//todo sharedSize zmienic na nowy sposob liczenia
	/*
	 agg_kernel_1<<<blocksPerGrid, threadsPerBlock, cacheSize>>>(numSamples, d_samples, cacheSize, d_aggr_min, d_aggr_max, d_aggr_avg);
	 //wywołanie kernela zbierającego dane z niezależnych bloków (zatem mamy tylko jeden blok)
	 threadsPerBlock = blocksPerGrid;
	 blocksPerGrid = 1;
	 cacheSize = threadsPerBlock * sizeof(float) * NUM_AGGREGATORS;
	 agg_kernel_2<<<blocksPerGrid, threadsPerBlock, cacheSize>>>(numSamples, cacheSize, d_aggr_min, d_aggr_max, d_aggr_avg);

	 */

	cudaDeviceSynchronize();
	CHECK_SINGLE_ERROR()
			;
	checkCudaErrors(cudaMemcpy(h_aggr_min, d_aggr_min, aggHeapSize, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_aggr_max, d_aggr_max, aggHeapSize, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_aggr_avg, d_aggr_avg, aggHeapSize, cudaMemcpyDeviceToHost));
#ifdef DEBUG
	printAggHeap(numSamples, h_aggr_min);
#else
	//TODO stdout binary output
#endif
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
#ifdef TEST
	process("data/Osoba_concat.txt", argc, argv);
//	process("Test_data.txt", argc, argv);
//	process("data/Osoba_2000linii.txt", argc, argv);
//	process("data/Osoba_1row.txt", argc, argv);
#else
	process("data/Osoba_concat.txt", argc, argv);
#endif
	CHECK_LAUNCH_ERROR()
				;
	printf("\n\nprocess ended.");
}
