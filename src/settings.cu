/*
 * settings.cu
 *
 *  Created on: 02-06-2015
 *      Author: adam
 */

#include "settings.h"
#include <stdio.h>
Settings initSettings(int argc, char** argv) {
	Settings s;
	if (argc <2) {
		s.GROUP_A_SIZE = SETTINGS_GROUP_A_SIZE;
		s.GROUP_B_SIZE = SETTINGS_GROUP_B_SIZE;
		s.GROUP_C_SIZE = SETTINGS_GROUP_C_SIZE;
		s.NUM_BLOCKS = SETTINGS_NUM_BLOCKS;
	}
	else {
		s.GROUP_A_SIZE = atoi(argv[1]);
		s.GROUP_B_SIZE = atoi(argv[2]);
		s.GROUP_C_SIZE = atoi(argv[3]);
		s.NUM_BLOCKS = atoi(argv[4]);
	}
	printf("\nCMD Settings: \n\tA threads : %d \n\tB threads : %d \n\tC threads : %d \n\tnum blocks: %d\n\n", s.GROUP_A_SIZE, s.GROUP_B_SIZE, s.GROUP_C_SIZE, s.NUM_BLOCKS);
	return s;
}

