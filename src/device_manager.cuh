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
struct BlockState {
	/*
	 * Podział pracy nad strumieniem:
	 * [read_start  - agg_start-1 ] // przedział do wczytania
	 * [agg_start - write_start-1 ] // przedział do obliczenia
	 * [write_start - write_end -1] // przedział do zapisania do heap
	 */
	int partReadIndex;
	int partAggIndex;
	int partWriteIndex;
	int num_A_threads;
	int num_B_threads;
	int num_C_threads;
	int numParts;
	int numPartsPerBlock;
	int partSize;
	int numSamples;
};

__device__ void setupBlockState(BlockState *state, int numSamples) {
	const int threadsPerBlock = blockDim.x;
	int numCountThreads = blockDim.x / (1 + GROUP_RATIO); //jeśli mamy 10 wątków to A=3 B=4 C=3
	if (numCountThreads == 0) {
		numCountThreads = 1;
	}
	int numMemThreads = (blockDim.x - numCountThreads) / 2;
	if (numMemThreads == 0) {
		numMemThreads = 1;
	}
	else if (numMemThreads > 32) {
		numMemThreads = (numCountThreads / 32) * 32;
		numCountThreads = blockDim.x - 2 * numMemThreads;
	}
	//
	state->numSamples = numSamples;
	//
	state->num_A_threads = numMemThreads;
	state->num_B_threads = threadsPerBlock - 2 * numMemThreads;
	state->num_C_threads = numMemThreads;

	state->partReadIndex = blockIdx.x * state->num_B_threads; //w pierwszej fazie trzeba ustawić na prawidłowy indeks pierwszej paczki dla bloku
	state->partAggIndex = -1; //w pierwszej fazie nie ma czego liczyć
	state->partWriteIndex = -1; //w pierwszej fazie nie  ma czego odsyłać do global
	//
	state->partSize = AGG_TEST_108;
	state->numParts = divceil(numSamples, state->partSize);
	state->numPartsPerBlock = divceil(state->numParts, blockDim.x);
}
__device__ void updateBlockState(int i, numSamples, BlockState* state) {
	//od jakiej paczki ma zacząć wczytywanie blok (zrobią to równolegle wątki A)
	int partIndexForBlock = state->num_B_threads * (i * blockDim.x + blockIdx.x);

	//to co grupa B obliczyła teraz ma być zapisane przez C
	state->partWriteIndex = state->partAggIndex;
	//to co grupa A wczytała teraz ma zostać obliczone przez B
	state->partAggIndex = state->partReadIndex;
	//grupa A wczytuje nowe paczki
	state->partReadIndex += blockDim.x * state->num_B_threads;

}

__device__ void thread_A_iter(int i, const int localId, const BlockState *state) {
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
	const int elementIndex;

	for (int i = 0; i < numElementsToCopyByThread; i++) {
		elementIndex = numCopyingThreads * i + localId;
		pritnf("\nT:[%d %d] A:%d i:%d", blockIdx.x, threadIdx.x, localId, i);
		__syncthreads();
	}

}
__device__ void thread_B_iter(const int i, const int localId, BlockState *state) {

}
__device__ void thread_C_iter(const int i, const int localId, BlockState *state) {

}

__global__ void kernel_manager(int numSamples, float *samples, AggrPointers *outPointers) {
	extern __shared__ void shared[];
	BlockState *state = (BlockState*) shared;
	const int blockIndex = blockIdx.x;

	if (threadIdx.x == 0) { //zainicjalizujmy stan
		setupBlockState(state);
	}

//wątki są przydzielone do grup w kolejności A, C, B
	const int firstThreadInCGroupIndex = state->num_A_threads;
	const int firstThreadInBGroupIndex = firstThreadInCGroupIndex + state->num_B_threads;
	const int localId_A = threadIdx.x;
	const int localId_C = threadIdx.x - firstThreadInCGroupIndex;
	const int localId_B = threadIdx.x - firstThreadInBGroupIndex;
	const int numIterations = state->numPartsPerBlock + 2; //potok ma 3 elementy zatem trzeba dodać 2 iteracje kończące
	for (int i = 0; i < numIterations; i++) { //ile będzie iteracji? Ile każdy blok wczyta porcji danych?

		if (localId_B >= 0) { //jest to wątek typu B

		}
		else if (localID_C >= 0) { //to jest wątek typu C

		}
		else { //wątek typu A
			thread_A_iter(i, localId_A, block);
		}
		__syncthreads();
		if (threadIdx.x == 0) {
			updateBlockState(i, state);
		}
		__syncthreads();

	}

}

#endif /* DEVICE_MANAGER_CUH_ */
