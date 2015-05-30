/*
 * device_manager.cuh
 *
 *  Created on: 27-05-2015
 *      Author: adam
 */

#ifndef DEVICE_MANAGER_CUH_
#define DEVICE_MANAGER_CUH_
#include "settings.h"
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
	float *samplesCache;
	int samplesCacheCount;
	AggrPointers *outPointers;
};

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

	// cache setup
	state->samples = samples;
	state->samplesCache = (float*) state + sizeof(BlockState);	//cache dla wczytanych sampli są zaraz za BlockState
	state->samplesCacheCount = state->partSize * state->num_B_threads;

}
__device__ void updateBlockState(int i, const int numSamples, BlockState* state) {

	//to co grupa B obliczyła teraz ma być zapisane przez C
	state->firstWritePart = state->firstCountPart;
	//to co grupa A wczytała teraz ma zostać obliczone przez B
	state->firstCountPart = state->firstReadPart;
	//grupa A wczytuje nowe paczki
	state->firstReadPart += gridDim.x * state->num_B_threads;

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
	tsync(if(localId==0)tlog("iter %d, starting part %d el %d",i, startingPart, startingElement));
	dbgi(numElementsToCopyByThread);
	for (int j = 0; j < numElementsToCopyByThread; j++) {
		localElementIndex = j * numCopyingThreads + localId;
		globalElementIndex = startingElement + localElementIndex;
		//załaduj element z global do cache
		state->samplesCache[localElementIndex] = state->samples[globalElementIndex];
		if (localElementIndex < numElementsToCopyByBlock && globalElementIndex < numSamples) {
			tlog("i: %d j:%d (%d -> %d) #%d %f=%f", i, j, globalElementIndex, localElementIndex, state->numSamples,state->samplesCache[localElementIndex], state->samples[globalElementIndex]);
		}

		__syncthreads();
	}

}
__device__ void thread_B_iter(const int i, const int localId, BlockState *state) {

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
	const int firstThreadInBGroupIndex = firstThreadInCGroupIndex + state->num_B_threads;
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

		}
		else if (threadIsC && i > 1) { //to jest wątek typu C

		}
		__syncthreads();
		if (threadIdx.x == 0) {
			updateBlockState(i, numSamples, state);
		}
		__syncthreads();

	}

}

#endif /* DEVICE_MANAGER_CUH_ */
