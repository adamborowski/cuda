/*
 * kernels.cuh
 *
 *  Created on: 20-05-2015
 *      Author: adam
 */

#include "Utils.h"
#include "CudaProj.h"
#include "device_aggregator.cuh"
#include "common_utils.cuh"

#ifndef KERNELS_CUH_
#define KERNELS_CUH_

__global__ void agg_kernel_1(int numSamples, float* samples, int cacheSize, float * d_aggr_min, float * d_aggr_max, float * d_aggr_avg) {

	const int globalID = blockDim.x * blockIdx.x + threadIdx.x;
	const int threadsPerBlock = blockDim.x;
#ifdef DEBUG
	if (globalID == 0) {
		int numThreads = blockDim.x * gridDim.x;
		printf("\nreal num threads: %d\n", numThreads);
	}
#endif
	extern __shared__ float shared[];
	float* s_min = shared;
	float* s_max = NEXT_ARRAY(s_min, threadsPerBlock);
	float* s_avg = NEXT_ARRAY(s_max, threadsPerBlock);
	cleanArray(threadsPerBlock * NUM_AGGREGATORS, shared);

	//
	AggrPointers globalInput = { samples, samples, samples };
	AggrPointers globalOutput = { d_aggr_min, d_aggr_max, d_aggr_avg };
	AggrPointers sharedInput = { s_min, s_max, s_avg };
	AggrPointers sharedOutput = { s_min, s_max, s_avg };
	int startingAggType = AGG_SAMPLE;

	//pierwsze musi pójść z globalu
	device_aggregate(numSamples, true, startingAggType, AGG_TEST_1, AGG_TEST_3, &globalInput, &sharedOutput, &globalOutput);
	device_aggregate(numSamples, false, startingAggType, AGG_TEST_3, AGG_TEST_6, &sharedInput, &sharedOutput, &globalOutput);
	device_aggregate(numSamples, false, startingAggType, AGG_TEST_6, AGG_TEST_18, &sharedInput, &sharedOutput, &globalOutput);
	if (globalID == 0) {
		printf("\n>>>>>> krenel ended <<<<<<<");
	}
}

__global__ void agg_kernel_2(int numSamples, int cacheSize, float * d_aggr_min, float * d_aggr_max, float * d_aggr_avg) {

	const int globalID = blockDim.x * blockIdx.x + threadIdx.x;
	const int threadsPerBlock = blockDim.x;
#ifdef DEBUG
	if (globalID == 0) {
		int numThreads = blockDim.x * gridDim.x;
		printf("\nreal num threads: %d\n", numThreads);
	}
#endif
	extern __shared__ float shared[];
	float* s_min = shared;
	float* s_max = NEXT_ARRAY(s_min, threadsPerBlock);
	float* s_avg = NEXT_ARRAY(s_max, threadsPerBlock);
	cleanArray(threadsPerBlock * NUM_AGGREGATORS, shared);

	/*
	 * Uwaga! aby kontynuować działanie z poprzedniego kernela skopiować
	 * ostatni wynik!!
	 * Trzeba zastanowić się którą częćś i jakiej
	 */

	//odnajdź offset ostatniej agregacji w pamięci globalnej, jest jedna pamięć shared bo jeden blok
	int offset = getAggOffset(numSamples, AGG_TEST_18);

	//
	AggrPointers globalInput = { &d_aggr_min[offset], &d_aggr_max[offset], &d_aggr_avg[offset] };
	AggrPointers sharedInput = { s_min, s_max, s_avg };
	AggrPointers globalOutput = { d_aggr_min, d_aggr_max, d_aggr_avg };
	AggrPointers sharedOutput = sharedInput;

	int startingAggType = AGG_TEST_18;

	//pierwsze musi pójść z globalu ale ma udawać shared - czyli uwzględnić offset
	device_aggregate(numSamples, false, startingAggType, AGG_TEST_18, AGG_TEST_36, &globalInput, &sharedOutput, &globalOutput);
	device_aggregate(numSamples, false, startingAggType, AGG_TEST_36, AGG_TEST_108, &sharedInput, &sharedOutput, &globalOutput);
	if (globalID == 0) {
		printf("\n>>>>>> krenel ended <<<<<<<");
	}
}

#endif /* KERNELS_CUH_ */
