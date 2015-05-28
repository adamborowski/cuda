/*
 * common_utils.cuh
 *
 *  Created on: 19-05-2015
 *      Author: adam
 */
#include <stdio.h>
#ifndef COMMON_UTILS_CUH_
#define COMMON_UTILS_CUH_

__host__ __device__
int divceil(int a, int b) {
	return (a + b - 1) / b;
}
__host__ __device__ void cleanArray(int count, float* array) {
	for (int i = 0; i < count; i++)
		array[i] = NAN;
}
/**
 * Calculates aggregation count for specified data size
 */
__host__ __device__ int getAggCount(const int numSamples, const int aggType) {
	return divceil(numSamples, aggType);
}
/**
 * Offset of tree heap. The second smallest aggregation is at the begining.
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
#ifndef TEST
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
#else
	switch (aggType) { //in this switch if any case will occur, we continue to each next case (don't break)
		case AGG_ALL:
		offset += getAggCount(numSamples, AGG_TEST_108);
		/* no break */
		case AGG_TEST_108:
		offset += getAggCount(numSamples, AGG_TEST_36);
		/* no break */
		case AGG_TEST_36:
		offset += getAggCount(numSamples, AGG_TEST_18);
		/* no break */
		case AGG_TEST_18:
		offset += getAggCount(numSamples, AGG_TEST_6);
		/* no break */
		case AGG_TEST_6:
		offset += getAggCount(numSamples, AGG_TEST_3);
		/* no break */
//		case AGG_TEST_3:
//		we dont to put samples so we shouldn't add offset for it
		/* no break */
//		case AGG_TEST_1:
//		we have nothing before to add offset
//		;
	}
#endif
	return offset;
}

__device__ __host__ int getWiderAggr(int aggr) {
#ifdef TEST
	switch(aggr) {
		case AGG_SAMPLE: return AGG_TEST_3;
		case AGG_TEST_3: return AGG_TEST_6;
		case AGG_TEST_6: return AGG_TEST_18;
		case AGG_TEST_18: return AGG_TEST_36;
		case AGG_TEST_36: return AGG_TEST_108;
		case AGG_TEST_108: return AGG_ALL;
	}
#else
	switch (aggr) {
	case AGG_SEC_1:
		return AGG_SEC_10;
	case AGG_SEC_10:
		return AGG_MIN;
	case AGG_MIN:
		return AGG_MIN_10;
	case AGG_MIN_10:
		return AGG_MIN_30;
	case AGG_MIN_30:
		return AGG_HOUR;
	case AGG_HOUR:
		return AGG_HOUR_24;
	case AGG_HOUR_24:
		return AGG_YEAR;
	case AGG_YEAR:
		return AGG_ALL;
	}
#endif
	return AGG_BAD;
}

/**
 * zwraca rozmiar kawałka danej tablicy uwzględniajc że ilość elementów nie jest wielokrotnością wielkości kawałka
 */
__device__ int getChunkSize(const int numItems, const int chunkSize, const int chunkIndex) {
	const int numFullChunks = numItems / chunkSize;
	if (numFullChunks * chunkSize == numItems) {
		//tablica idealnie wyrównana
		if (chunkIndex < numFullChunks) {
			return chunkSize;
		}
		return BAD_CHUNK;
	}
	//tablica niewyrównana
	if (chunkIndex < numFullChunks) {
		//pełny chunk
		return chunkSize;
	}
	if (chunkIndex == numFullChunks) {
		//jest to ostatni kawałeczek
		return numItems % chunkSize;
	} else if (chunkIndex > numFullChunks) {
		return BAD_CHUNK;
	}
	return BAD_CHUNK;
}
/*
 * Zwraca globalny nr generowanego kawałka dla policzenia przez dany wątek
 * TODO obecnie ustala się, że samplesPerBlock = blockDim.x/aggType ale
 * można w przyszłości znieść to ograniczenie i używać nowej zmiennej
 * >>> zaokr.dol(blockDim.x,maxKernelAgg)/aggType - czyli uwzględniając maksymalną agregację do policzenia w kernelu
 */
__device__ int mapThreadToGlobalChunkIndex(int aggType) {
	int chunksPerBlock = divceil(blockDim.x , aggType);
	//ile tego typu danych mamy w bloku?
	if (threadIdx.x >= chunksPerBlock) {
		//ten wątek nie ma już co liczyć w tym bloku
		return BAD_CHUNK;
	}
	return chunksPerBlock * blockIdx.x + threadIdx.x;
}


void printHeap(const int size, float* heap) {
	int prevAggType = AGG_SAMPLE;
	int currentAggType = getWiderAggr(prevAggType);
	while (currentAggType != AGG_ALL) {
		printf("\n    %d->%d", prevAggType, currentAggType);
		int aggOffset = getAggOffset(size, currentAggType);
		int nextOffset = getAggOffset(size, getWiderAggr(currentAggType));
		int count = nextOffset - aggOffset;
		for (int i = 0; i < count; i++) {
			printf("\n\t\t[%d] = % 3.6f", i, heap[aggOffset + i]);
		}
		//
		prevAggType=currentAggType;
		currentAggType = getWiderAggr(currentAggType);
	}
	printf("\n-------------------------------------------");
}



#endif /* COMMON_UTILS_CUH_ */
