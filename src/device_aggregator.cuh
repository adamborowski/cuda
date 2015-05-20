/*
 * device_aggregator.h
 *
 *  Created on: 19-05-2015
 *      Author: adam
 */

#ifndef DEVICE_AGGREGATOR_H_
#define DEVICE_AGGREGATOR_H_

#include "device_aggregator.cuh"
#include "common_utils.cuh"
#include <helper_cuda.h>

#ifdef DEBUG

#define test(tmp, out) if (*out!=tmp) printf("\nFailed to write by localId = %d.\t%f!=%f. &out = %p.", threadIdx.x, *out, tmp, out);
#else
#define test(localId, tmp, out)
#endif
struct AggrPointers {
	float* min;
	float* max;
	float* avg;
};
struct AggrValues {
	float min;
	float max;
	float avg;
};
/**
 * Policzenie agregacji min max avg z krokiem
 * Wykonuje to jeden wątek
 */
__device__ void device_count_aggregation(const int stepSize, AggrPointers input, AggrValues* output) {
	float min = INFINITY;
	float max = -1 * INFINITY;
	float avg = 0;
	int i;
	for (i = 0; i < stepSize; i++) {
		//min aggregation
		if (input.min[i] < min) {
			min = input.min[i];
		}
		//max aggregation
		if (input.max[i] > max) {
			max = input.max[i];
		}
		//avg aggregation
		avg += input.avg[i]; //collect sum of input
	}
	avg /= (float) stepSize; //convert sum to average
	//dopiero teraz wpisujmy wynik, bo nie wiemy czy output jest w szybkiej czy wolnej pamięci
	output->min = min;
	output->max = max;
	output->avg = avg;
	//testuj zapis
	test(min, &output->min);test(max, &output->max);test(avg, &output->avg);
}

/*
 * @param bufferOffset - dodatkowa zmienna oznaczająca offset buforów wejściowych - przydatne gdy mamy
 * pamięć globalną i shared, wtedy offset jest inaczej liczony
 * @param outputSpec wskaźniki na tablice w shared memory
 * @param globalSpec wskaźniki na bufory wynikowe w pamięci globalnej
 */
__device__ void device_aggregate(
		int numSamples, bool inputIsGlobal, int startingAggType,
		int inputAggType, int outputAggType,
		AggrPointers* inputSpec, AggrPointers* sharedOutput, AggrPointers* globalOutput
		) {

	AggrValues aggResult;
	int threadLocalId = threadIdx.x;
	int chunkSize = outputAggType / inputAggType;
	int inputIndex, inputChunkIndex;

	int mappedChunkIndex = mapThreadToGlobalChunkIndex(outputAggType / startingAggType);

	if (inputAggType >= 18) {
		printf("\n%d->%d thread [%d,%d] mappedChunkIndex: %d", inputAggType, outputAggType, blockIdx.x, threadIdx.x, mappedChunkIndex);
	}
	int realChunkSize = 0;
	if (mappedChunkIndex != BAD_CHUNK) {	//nasz wątek w bloku ma co robić (nie chodzi o wyrównanie danych tylko o to że w każdym bloku agregacje są liczone tylko z bloków
		if (inputIsGlobal) {	//wiemy, że jest to odczyt z pamięci globalnej
			inputChunkIndex = mappedChunkIndex;	//każdy wątek bierze z globalnej tablicy swój chunk
		} else {	//wiemy że jest to odczyt z pamięci shared
			inputChunkIndex = threadLocalId;	//każdy wątek w bloku bierze swój chunk z tablicy shared w tym bloku
		}
		inputIndex = inputChunkIndex * chunkSize;	//wskazanie na pierwszy element chunku
		int numInputElements = divceil(numSamples, inputAggType);	//ile w ogóle jest elementów do zagregowania
		realChunkSize = getChunkSize(numInputElements, chunkSize, mappedChunkIndex);
	}

	if (realChunkSize > 0) { //jeśli wątek ma co robić
		printf("\n%d->%d inputindex: %d, thread [%d.%d], chunkSize: %d (real %d). mapped chunk: %d", inputAggType, outputAggType, inputIndex, blockIdx.x, threadIdx.x, chunkSize, realChunkSize, mappedChunkIndex);
		AggrPointers inputPointers = { &inputSpec->min[inputIndex], &inputSpec->max[inputIndex], &inputSpec->avg[inputIndex] };
		device_count_aggregation(realChunkSize, inputPointers, &aggResult);
		//also write to global memory as a result of AGG_10s
		int outputOffset = getAggOffset(numSamples, outputAggType); //we got offset to write 10s aggreations
		int globalPtr = outputOffset + mappedChunkIndex; //gdzie nasza próbka wyląduje w globalu?
		//XXX prawdopodobnie zapis do pamięci globalnej spowoduje że te wątki zostaną
		//XXX zserializowane i inne w tym czasie sie policzą
		globalOutput->min[globalPtr] = aggResult.min;
		globalOutput->max[globalPtr] = aggResult.max;
		globalOutput->avg[globalPtr] = aggResult.avg;
	}
	__syncthreads(); //poczekaj aż każdy odczyta bufor
	if (realChunkSize > 0) { //wątek miał co liczyć?
		//teraz już każdy ma swój aggResult, pora zacząć (jak wiemy...) nadpisywać cache :)
//		printf("\n________________ %f", aggResult.min);
		sharedOutput->min[threadLocalId] = aggResult.min;
		sharedOutput->max[threadLocalId] = aggResult.max;
		sharedOutput->avg[threadLocalId] = aggResult.avg;
	}
	__syncthreads();		//zapewnij, że dane będą spójnie zapisane w sharedOutput();
}

#endif /* DEVICE_AGGREGATOR_H_ */
