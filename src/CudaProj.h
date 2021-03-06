/*
 * CudaProj.h
 *
 *  Created on: 12-05-2015
 *      Author: adam
 */

#ifndef CUDAPROJ_H_
#define CUDAPROJ_H_

#include "settings.h"


#define AGG_SAMPLE 1
#define AGG_ALL 0 // special value used in calculating offset of all heap
#define AGG_BAD -1
#ifndef TEST

#define AGG_SEC_1 1
#define AGG_SEC_10 10
#define AGG_MIN 60
#define AGG_MIN_10 AGG_MIN*10
#define AGG_MIN_30 AGG_MIN*30
#define AGG_HOUR AGG_MIN*60
#define AGG_HOUR_24 AGG_HOUR*24
#define AGG_YEAR AGG_HOUR_24*365

#else


#define AGG_TEST_1 1
#define AGG_TEST_3 3
#define AGG_TEST_6 6
#define AGG_TEST_18 18
#define AGG_TEST_36 36
#define AGG_TEST_108 108

#endif

#define BAD_CHUNK -1
#define NUM_AGGREGATORS 3 // MIN, MAX, AVG
#endif /* CUDAPROJ_H_ */
