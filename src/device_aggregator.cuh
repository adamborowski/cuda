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

#define test(tmp, out) if (*out!=tmp) printf("\nFailed to write by threadId = %d.\t%f!=%f. &out = %p.", threadIdx.x, *out, tmp, out);
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
 * Przygotowuje i wykonuje algorytm agregacji dla danego wątka w bloku
 * @param numSamples ile sampli jest w programie
 * @param inputIsGlobal definiuje czy indeks ma być obliczny z uwzględnieniem odrębnej czy wspólnej pamięci
 * @param startingAggType jakiej agregacji odpowiada jeden wątek w bloku (każdy kernel inaczej tworzy bloki stąd to rozróżnienie przy obliczaniu indeksów)
 * @param inputAggType z jakiej agregacji liczmy nową
 * @param outputAggType jaką agregację liczymy
 * @param inputSpec wskaźniki danych wejściowych
 * @param sharedOutput wskaźniki do pamięci shared dla danych wyjściowych
 * @param globalOutput wskaźniki do stert globalnych dla danych wyjściowych
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

	int realChunkSize = 0;
	if (mappedChunkIndex != BAD_CHUNK) {	//nasz wątek w bloku ma co robić (nie chodzi o wyrównanie danych tylko o to że w każdym bloku agregacje są liczone tylko z bloków, więc mogą występować dziury
		//TODO zamiast inputIsGlobal dostarczać do funkcji przesunięte wskaźniki (shared vs global)
		if (inputIsGlobal) {	//wiemy, że jest to odczyt z pamięci globalnej
			inputChunkIndex = mappedChunkIndex;	//każdy wątek bierze z globalnej tablicy swój chunk
		} else {	//wiemy że jest to odczyt z pamięci shared
			inputChunkIndex = threadLocalId;	//każdy wątek w bloku bierze swój chunk z tablicy shared w tym bloku
		}
		inputIndex = inputChunkIndex * chunkSize;	//wskazanie na pierwszy element chunku
		int numInputElements = divceil(numSamples, inputAggType);	//ile w ogóle jest elementów do zagregowania globalnie
		realChunkSize = getChunkSize(numInputElements, chunkSize, mappedChunkIndex);	//ile jest elementów do zagregowania dla tego wątku (uwzgl. padding)
	}

	if (realChunkSize > 0) { //jeśli wątek ma co robić
		printf("\n%3d->%-3d inputIndex: %4d thread [%3d.%-3d] chunkSize: %d->%d mappedChunk: %d", inputAggType, outputAggType, inputIndex, blockIdx.x, threadIdx.x, chunkSize, realChunkSize, mappedChunkIndex);
		AggrPointers inputPointers = { &inputSpec->min[inputIndex], &inputSpec->max[inputIndex], &inputSpec->avg[inputIndex] };
		device_count_aggregation(realChunkSize, inputPointers, &aggResult);
		int outputOffset = getAggOffset(numSamples, outputAggType); //offset na stercie dla tego typu agregacji
		int globalPtr = outputOffset + mappedChunkIndex; //gdzie nasza próbka wyląduje na stercie globalnej?
		//XXX prawdopodobnie zapis do pamięci globalnej spowoduje że te wątki zostaną
		//XXX zserializowane i inne w tym czasie sie policzą, zatem tutaj jest pewna optymalizacja wykonana
		globalOutput->min[globalPtr] = aggResult.min;
		globalOutput->max[globalPtr] = aggResult.max;
		globalOutput->avg[globalPtr] = aggResult.avg;
	}
	__syncthreads(); //poczekaj aż nikt nie będzie potrzebował shared memory
	if (realChunkSize > 0) { //wątek miał co liczyć?
		//teraz już każdy ma swój aggResult, pora zacząć (jak wiemy...) nadpisywać cache :)
		sharedOutput->min[threadLocalId] = aggResult.min;
		sharedOutput->max[threadLocalId] = aggResult.max;
		sharedOutput->avg[threadLocalId] = aggResult.avg;
	}
	__syncthreads();		//zapewnij, że dane będą spójnie zapisane w sharedOutput();
}

#endif /* DEVICE_AGGREGATOR_H_ */
