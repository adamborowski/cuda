/*
 * device_manager.cuh
 *
 *  Created on: 27-05-2015
 *      Author: adam
 */

#ifndef DEVICE_MANAGER_CUH_
#define DEVICE_MANAGER_CUH_
#include "settings.h"
#include "utils.h"
#include "CudaProj.h"
#include "common_utils.cuh"
struct BlockState {
	/*
	 * Podział pracy nad strumieniem:
	 * [read_start  - agg_start-1 ] // przedział do wczytania
	 * [agg_start - write_start-1 ] // przedział do obliczenia
	 * [write_start - write_end -1] // przedział do zapisania do heap
	 */
	int firstReadPart;
	int firstCountPart;
	int firstWritePart;
	int num_A_threads;
	int num_B_threads;
	int num_C_threads;
	int numParts;
	int numPartsPerBlock;
	int partSize;
	int numSamples;
	float * samples;
	float *heapCacheAC; //cache sterty do wczytywania z global (A) i zapisu do global (C)
	float *heapCacheB; //cache sterty do obliczeń przez wątki B
	int heapCacheCount;
	int heapCacheSamplesCount;
	AggrPointers outPointers;
	Settings settings;
};
/*
 * @return ile bajtów zajmuje shared memory
 */
__device__ __host__ long initializeSharedMemory(BlockState *state) {
	const int samplesCount = state->partSize * state->num_B_threads;
	const int heapCount = getAggOffset(samplesCount, AGG_ALL);
	//
	float* movingPtr = (float*) state + sizeof(BlockState);
	const int SIZE = sizeof(float);
	//
	state->heapCacheCount = heapCount;
	state->heapCacheSamplesCount = samplesCount;
	//alokuj cache AC
	state->heapCacheAC = movingPtr;
	movingPtr += heapCount * SIZE;
	//alokuj cache B
	state->heapCacheB = movingPtr;
	movingPtr += heapCount * SIZE;
	//
	return (long) movingPtr - (long) state;
}

__device__ void setupBlockState(BlockState *state, int numSamples, float *samples, AggrPointers outPointers) {
	const int threadsPerBlock = blockDim.x;
	int numCountThreads = threadsPerBlock - (state->settings.GROUP_A_SIZE + state->settings.GROUP_C_SIZE); //jeśli mamy 10 wątków to A=3 B=4 C=3
	//
	state->numSamples = numSamples;
	state->partSize = AGG_TEST_108;
	//
	state->num_A_threads = state->settings.GROUP_A_SIZE;
	state->num_B_threads = numCountThreads;
	state->num_C_threads = state->settings.GROUP_C_SIZE;

	state->firstReadPart = blockIdx.x * state->num_B_threads; //w pierwszej fazie trzeba ustawić na prawidłowy indeks pierwszej paczki dla bloku
	state->firstCountPart = -1; //w pierwszej fazie nie ma czego liczyć
	state->firstWritePart = -1; //w pierwszej fazie nie  ma czego odsyłać do global
	//

	state->numParts = divceil(numSamples, state->partSize);

	state->numPartsPerBlock = divceil(state->numParts, gridDim.x);

	state->samples = samples;
	state->outPointers = outPointers;
	initializeSharedMemory(state);

}
__device__ void updateBlockState(int i, const int numSamples, BlockState* state) {

	//to co grupa B obliczyła teraz ma być zapisane przez C
	state->firstWritePart = state->firstCountPart;
	//to co grupa A wczytała teraz ma zostać obliczone przez B
	state->firstCountPart = state->firstReadPart;
	//grupa A wczytuje nowe paczki
	state->firstReadPart += gridDim.x * state->num_B_threads;
	//wymieniamy bufory sterty AC z B
	float* tmp = state->heapCacheAC;
	state->heapCacheAC = state->heapCacheB;
	state->heapCacheB = tmp;

}

__device__ void thread_A_iter(const int i, const int numIterations, const int localId, const BlockState *state) {
	//ile elementów trzeba skopiować w bloku w jednej iteracji?
	const int numElementsToCopyByBlock = state->num_B_threads * state->partSize;
	const int numCopyingThreads = state->num_A_threads;

	//czy jest co kopiowac?
	if (state->firstReadPart >= state->numParts) {
		return;
	}

	//w tej iteracji grupa A w bloku kopiuje ileś partii
	const int startingElement = state->firstReadPart * state->partSize;

#ifdef DEBUG
	if (localId == 0)
	tlog("A: iter %d, starting part %d el %d", i, state->firstReadPart, startingElement);
	cleanArray(getHeapOffset(numElementsToCopyByBlock, AGG_ALL), state->heapCacheAC);
#endif

	parallelCopy(
			localId, numCopyingThreads, numElementsToCopyByBlock,
			&state->samples[startingElement], state->heapCacheAC);
}
/*
 * Wykonuje agregację dla zakresu elementów
 * zapewnia obliczenie poprawnego elementu wejściowego i wyjściowego sterty (w zależności od nr wątku)
 */
__device__ void thread_B_aggregate(const int localId, const int inputAggr, const int outputAggr, const BlockState *state) {
	//skoro masz obliczyć jakąś agregację na pewnym zakresie, oblicz zakres wejściowy
	//indeks początkowy w pamięci shared to heapOffset(inputAggr)+ileInputPozeraWatek

	//czy jest co kopiowac?
	if (state->firstCountPart >= state->numParts) {
		return;
	}

	const int numInputElements = state->partSize / inputAggr;	//ile w bloku mamy bloków wejściowych
	const int numOutputElements = state->partSize / outputAggr;
	const int aggChunkSize = outputAggr / inputAggr;	//ile elementów jest agregowanych w jedno
	const int inputOffset = getHeapOffset(state->heapCacheSamplesCount, inputAggr) + localId * numInputElements;
	const int outputOffset = getHeapOffset(state->heapCacheSamplesCount, outputAggr) + localId * numOutputElements;
	AggrPointers input;
	AggrValues output;
	for (int i = 0; i < numOutputElements; i++) {
		input.min = &state->heapCacheB[inputOffset + i * aggChunkSize];
		input.max = input.min;	//TODO uproszczenie
		input.avg = input.min;	//TODO uproszczenie
		device_count_aggregation(aggChunkSize, input, &output);
		//co z align?
		state->heapCacheB[outputOffset + i] = output.min;
	}
}
__device__ void thread_B_iter(const int i, const int numIterations, const int localId, const BlockState *state) {
#ifdef DEBUG
	if (localId == 0)
	tlog("B: iter %d", i);
#endif

	thread_B_aggregate(localId, AGG_SAMPLE, AGG_TEST_3, state);
	thread_B_aggregate(localId, AGG_TEST_3, AGG_TEST_6, state);
	thread_B_aggregate(localId, AGG_TEST_6, AGG_TEST_18, state);
	thread_B_aggregate(localId, AGG_TEST_18, AGG_TEST_36, state);
	thread_B_aggregate(localId, AGG_TEST_36, AGG_TEST_108, state);
	//XXX teraz w heapie będą się minimzalizowały indeksy dlatego widać że działa skoro AGG_3 wyliczył z (0,1,2)->0, (3,4,5)->3 i wygląda tak: 0,3,6,9,...
}
__device__ void thread_C_copyLevel(const int localId, const int inputAggr, const int outputAggr, BlockState* state) {
	//offset lokalny zależy tylko od stopnia agregacji (sterta nie jest wspóldzielona)
	const int localOffset = getHeapOffset(state->heapCacheSamplesCount, outputAggr);
	//offset lokalny zależy od tego, ile było iteracji głównych i ile pozostałe bloki obliczyły
	const int globalAggOffset = getAggOffset(state->numSamples, outputAggr);	//globalnie agregacja zaczyna się od tej pozycji
	const int numAggInPart = state->partSize / outputAggr;	//ile tego typu agregacji mieści sie w pojedynczej partii
	const int numLocalAggChunks = state->num_B_threads * numAggInPart;	//ile w tym bloku wyliczono elementów danej agregacji
	const int firstWriteAgg = state->firstWritePart * numAggInPart;	//globalny indeks pierwszej agregacji tego typu w tym bloku
	const int globalDestination = globalAggOffset + firstWriteAgg;

	//czy jest co kopiowac?
	if (state->firstWritePart >= state->numParts) {
		return;
	}

	//tutaj trzeba zabezpieczyć aby nie wyjść poza faktyczny rozmiar
	//ile w bloku będziemy mieli elementów do skopiowania?
	//ile elementów przyjmie global?
	const int globalFirstChunkIndex = firstWriteAgg;
	const int globalNumAgg = getAggCount(state->numSamples, outputAggr);

	int dataLength;
	if (globalFirstChunkIndex + numLocalAggChunks < globalNumAgg) {
		dataLength = numLocalAggChunks;
	}
	else {
		dataLength = globalNumAgg - globalFirstChunkIndex;
//		tlog("data length: %d", dataLength);
	}
//	tlog("firstWriteAgg: %d, fixed: %d real: %d", firstWriteAgg, numLocalAggChunks, realNumLocalAggChunks);
	parallelCopy(localId, state->num_C_threads, dataLength, &state->heapCacheAC[localOffset], &state->outPointers.min[globalDestination]);
}
__device__ void thread_C_iter(const int i, const int localId, BlockState *state) {
#ifdef DEBUG
	if (localId == 0)
	tlog("C: iter %d", i);
#endif
	/*
	 * zadanie polega na skopiowaniu równoległym sterty shared piętro po piętrze ponieważ trzeba wtasować się w wyniki z innych bloków i iteracji
	 *
	 */
	thread_C_copyLevel(localId, AGG_SAMPLE, AGG_TEST_3, state);
	thread_C_copyLevel(localId, AGG_TEST_3, AGG_TEST_6, state);
	thread_C_copyLevel(localId, AGG_TEST_6, AGG_TEST_18, state);
	thread_C_copyLevel(localId, AGG_TEST_18, AGG_TEST_36, state);
	thread_C_copyLevel(localId, AGG_TEST_36, AGG_TEST_108, state);
}

__global__ void kernel_manager(Settings settings, int numSamples, float *samples, AggrPointers outPointers) {
	extern __shared__ float shared[];
	BlockState *state = (BlockState*) shared;
	if (threadIdx.x == 0) { //zainicjalizujmy stan
		state->settings = settings;
		setupBlockState(state, numSamples, samples, outPointers);
	}
	__syncthreads();

//wątki są przydzielone do grup w kolejności A, C, B
	const int firstThreadInCGroupIndex = state->num_A_threads;
	const int firstThreadInBGroupIndex = state->num_A_threads + state->num_C_threads;
	const int localId_A = threadIdx.x;
	const int localId_C = threadIdx.x - firstThreadInCGroupIndex;
	const int localId_B = threadIdx.x - firstThreadInBGroupIndex;
	const bool threadIsB = localId_B >= 0;
	const bool threadIsC = !threadIsB && localId_C >= 0;
	const bool threadIsA = !threadIsB && !threadIsC;
	const int numIterations = divceil(state->numPartsPerBlock, state->num_B_threads) + 2; //potok ma 3 elementy zatem trzeba dodać 2 iteracje kończące
	for (int i = 0; i < numIterations; i++) { //ile będzie iteracji? Ile każdy blok wczyta porcji danych?
		if (threadIsA && i + 2 < numIterations) { //wątek typu A
			thread_A_iter(i, numIterations, localId_A, state);
		}
		if (threadIsB && i > 0 && i + 1 < numIterations) { //jest to wątek typu B
			thread_B_iter(i, numIterations, localId_B, state);
		}
		else if (threadIsC && i > 1) { //to jest wątek typu C
			thread_C_iter(i, localId_C, state);
		}
		__syncthreads();
//		if (threadIdx.x == 0 && i > 0) {
//			printHeap(state->heapCacheSamplesCount, state->heapCacheB);
//		}
		if (threadIdx.x == 0) {
			updateBlockState(i, numSamples, state);
		}
		__syncthreads();
//		if (i == 2)
//			break; //FIXME REMOVE

	}

}

#endif /* DEVICE_MANAGER_CUH_ */
