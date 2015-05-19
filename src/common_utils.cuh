/*
 * common_utils.cuh
 *
 *  Created on: 19-05-2015
 *      Author: adam
 */

#ifndef COMMON_UTILS_CUH_
#define COMMON_UTILS_CUH_

__host__ __device__
int divceil(int a, int b) {
	return (a + b - 1) / b;
}
/**
 * Calculates aggregation count for specified data size
 */
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
	return -1;
}

#endif /* COMMON_UTILS_CUH_ */
