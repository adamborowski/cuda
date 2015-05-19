/*
 * device_aggregator.h
 *
 *  Created on: 19-05-2015
 *      Author: adam
 */

#ifndef DEVICE_AGGREGATOR_H_
#define DEVICE_AGGREGATOR_H_

#include "device_aggregator.cuh"
#include <helper_cuda.h>

#ifdef DEBUG

#define test(tmp, out) if (*out!=tmp) printf("\nFailed to write by localId = %d.\t%f!=%f. &out = %p.", threadIdx.x, *out, tmp, out);
#else
#define test(localId, tmp, out)
#endif
/**
 * Policzenie agregacji min max avg z krokiem
 * Wykonuje to jeden wątek
 */
__device__ void device_count_aggregation(const int stepSize, const float* minInput, const float* maxInput, const float* avgInput, float* minOutput, float* maxOutput, float* avgOutput) {
	float min = INFINITY;
	float max = -1 * INFINITY;
	float avg = 0;
	int i;
	for (i = 0; i < stepSize; i++) {
		//min aggregation
		if (minInput[i] < min) {
			min = minInput[i];
		}
		//max aggregation
		if (maxInput[i] > max) {
			max = maxInput[i];
		}
		//avg aggregation
		avg += avgInput[i]; //collect sum of input
	}
	avg /= (float) stepSize;//convert sum to average
	//dopiero teraz wpisujmy wynik, bo nie wiemy czy output jest w szybkiej czy wolnej pamięci
	*minOutput = min;
	*maxOutput = max;
	*avgOutput = avg;
	//testuj zapis
	test(min, minOutput);test(max, maxOutput);test(avg, avgOutput);
}

/*
 * @param minOutputBuffer - wskaźnik do całej tablicy globalnej
 */
__device__ void device_aggregate(
		int globalId, int aggType,
		float* minInput, float* maxInput, float *aggInput,
		float* minOutputBuffer, float* maxOutputBuffer, float* avgOutputBuffer
		) {

}

#endif /* DEVICE_AGGREGATOR_H_ */
