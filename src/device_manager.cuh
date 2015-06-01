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
	AggrPointers *outPointers;
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

__device__ void setupBlockState(BlockState *state, int numSamples, float *samples) {
	const int threadsPerBlock = blockDim.x;
	int numCountThreads = threadsPerBlock - (SETTINGS_GROUP_A_SIZE + SETTINGS_GROUP_C_SIZE); //jeśli mamy 10 wątków to A=3 B=4 C=3
	//
	state->numSamples = numSamples;
	state->partSize = AGG_TEST_108;
	//
	state->num_A_threads = SETTINGS_GROUP_A_SIZE;
	state->num_B_threads = numCountThreads;
	state->num_C_threads = SETTINGS_GROUP_C_SIZE;

	state->firstReadPart = blockIdx.x * state->num_B_threads; //w pierwszej fazie trzeba ustawić na prawidłowy indeks pierwszej paczki dla bloku
	state->firstCountPart = -1; //w pierwszej fazie nie ma czego liczyć
	state->firstWritePart = -1; //w pierwszej fazie nie  ma czego odsyłać do global
	//

	state->numParts = divceil(numSamples, state->partSize);

	state->numPartsPerBlock = divceil(state->numParts, gridDim.x);

	state->samples = samples;
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
	/*
	 * TODO przenieść zmienne lokalne iteracji do struktury która by była definiowana raz przez init_thread_A
	 * zrobić to na etap III
	 */
	/*
	 * 0. ile partii trzeba przeczytać z global (w ramach grupy A)
	 * 1. ile elementów tablicy trzeba skopiować w sumie?
	 * 2. ile iteracji wymagać będzie skopiowanie
	 * 3. w konkretnej iteracji jaki wątek jaki element ma skopiować?
	 */
	//ile elementów trzeba skopiować w bloku w jednej iteracji?
	const int numElementsToCopyByBlock = state->num_B_threads * state->partSize;
	//ile wątków kopiuje?
	const int numCopyingThreads = state->num_A_threads;
	//ile elementów ma skopiować pojedyczny wątek grupy A? Tyle będzie iteracji?
	const int numElementsToCopyByThread = divceil(numElementsToCopyByBlock, numCopyingThreads);
	//wsaźnik elementu w iteracji
	int globalElementIndex, localElementIndex;

	//w tej iteracji grupa A w bloku kopiuje ileś partii
	const int startingPart = state->firstReadPart;
	const int startingElement = startingPart * state->partSize;
	const int numSamples = state->numSamples;
	if (localId == 0)
		tlog("A: iter %d, starting part %d el %d", i, startingPart, startingElement);
//	dbgi(numElementsToCopyByThread);
	cleanArray(getHeapOffset(numElementsToCopyByBlock, AGG_ALL), state->heapCacheAC);
	for (int j = 0; j < numElementsToCopyByThread; j++) {
		localElementIndex = j * numCopyingThreads + localId;
		globalElementIndex = startingElement + localElementIndex;
		//załaduj element z global do cache AC
		if (localElementIndex < numElementsToCopyByBlock && globalElementIndex < numSamples) {
//			state->heapCacheAC[localElementIndex] = state->samples[globalElementIndex];
			state->heapCacheAC[localElementIndex] = globalElementIndex;	//Test
//			tlog("i: %d j:%d (%d -> %d) #%d %f=%f", i, j, globalElementIndex, localElementIndex, state->numSamples, state->heapCacheAC[localElementIndex], state->samples[globalElementIndex]);
		}

	}
//	if(blockIdx.x==0 && threadIdx.x==0 && i==0){
//		printHeap(state->heapCacheSamplesCount, state->heapCacheAC);
//	}

}
/*
 * Wykonuje agregację dla zakresu elementów
 * zapewnia obliczenie poprawnego elementu wejściowego i wyjściowego sterty (w zależności od nr wątku)
 */
__device__ void thread_B_aggregate(const int inputAggr, const int outputAggr, const BlockState *state){

}
__device__ void thread_B_iter(const int i, const int numIterations, const int localId, const BlockState *state) {
	if (localId == 0)
		tlog("B: iter %d", i);

	/*
	 * w tym wątku bierzemy jeden part i robimy dla niego agregacje i wysyłamy pod wskazany adres
	 * adres wyznaczamy: getHeapOffset(agg)+localId*getAggCount(partSize, aggr)
	 */
	const int myNumSamples = state->partSize;
	AggrPointers input;
	AggrValues output;
	const int inputAggr = AGG_SAMPLE;
	const int outputAggr = AGG_TEST_3;
	const int stepSize = outputAggr / inputAggr;	//z ilu elementów robi się agregację?
	const int numInputElementsPerThread = getAggCount(myNumSamples, inputAggr); //ile elementów pożera jeden wątek?
	const int numOutputElementsPerThread = getAggCount(myNumSamples, outputAggr); //ile elementów wypluwa jeden wątek?
	const int sharedHeapInputOffset = getHeapOffset(state->heapCacheSamplesCount, inputAggr);
	const int sharedHeapOutputOffset = getHeapOffset(state->heapCacheSamplesCount, outputAggr);
	const int myInputOffset = sharedHeapInputOffset + localId * numInputElementsPerThread;
	const int myOutputOffset = sharedHeapOutputOffset + localId * numOutputElementsPerThread;
	//XXX założenie - tylko MIN (jak będzie działać, doda się max i avg)

	//iteracja 1 -> 3
	const int numOutputChunks = getAggCount(myNumSamples, outputAggr);
	tlog("localId: %d, %d -> %d", localId, myInputOffset, myOutputOffset);
	for (int j = 0; j < numOutputChunks; j++) {
		input.min = &state->heapCacheB[myInputOffset + j * stepSize];
		input.max = &state->heapCacheB[myInputOffset + j * stepSize];
		input.avg = &state->heapCacheB[myInputOffset + j * stepSize];
		device_count_aggregation(stepSize, input, &output);
		state->heapCacheB[myOutputOffset + j] = output.min;
	}
	//TODO teraz w heapie będą się minimzalizowały indeksy dlatego widać że działa skoro AGG_3 wyliczył z (0,1,2)->0, (3,4,5)->3 i wygląda tak: 0,3,6,9,...
	tlog("B-> cacheAC: %p", state->heapCacheB);

}
__device__ void thread_C_iter(const int i, const int localId, BlockState *state) {

}

__global__ void kernel_manager(int numSamples, float *samples, AggrPointers *outPointers) {
	extern __shared__ float shared[];
	BlockState *state = (BlockState*) shared;
	if (threadIdx.x == 0) { //zainicjalizujmy stan
		setupBlockState(state, numSamples, samples);
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

		}
		__syncthreads();
		if (threadIdx.x == 0 && i > 0) {
			printHeap(state->heapCacheSamplesCount, state->heapCacheB);
		}
		if (threadIdx.x == 0) {
			updateBlockState(i, numSamples, state);
		}
		__syncthreads();
		if (i == 1)
			break; //FIXME REMOVE

	}

}

#endif /* DEVICE_MANAGER_CUH_ */
