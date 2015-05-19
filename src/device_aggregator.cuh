/*
 * device_aggregator.h
 *
 *  Created on: 19-05-2015
 *      Author: adam
 */

#ifndef DEVICE_AGGREGATOR_H_
#define DEVICE_AGGREGATOR_H_

#include "device_aggregator.cuh"
/**
 * Policzenie agregacji min max avg z krokiem
 * Wykonuje to jeden wątek
 */
__device__ void device_aggregate(const int count, const int stepSize, const float* minInput, const float* maxInput, const float* avgInput, float* minOutput, float* maxOutput, float* avgOutput) {
	float min = INFINITY;
	float max = -1 * INFINITY;
	float avg = 0;
	float i;
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
	avg /= (float) stepSize;
	//dopiero teraz wpisujmy wynik, bo nie wiemy czy output jest w szybkiej czy wolnej pamięci
	*minOutput = min;
	*maxOutput = max;
	*avgOutput = avg;
}

#endif /* DEVICE_AGGREGATOR_H_ */
