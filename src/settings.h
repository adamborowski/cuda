/*
 * settings.h
 *
 *  Created on: 20-05-2015
 *      Author: adam
 */

#ifndef SETTINGS_H_
#define SETTINGS_H_

#define TEST
//#define DEBUG
#define SETTINGS_GROUP_A_SIZE 32
#define SETTINGS_GROUP_B_SIZE 3
#define SETTINGS_GROUP_C_SIZE 32
#define SETTINGS_NUM_BLOCKS 4 //FIXME SET 2 or more
struct Settings {
	int GROUP_A_SIZE;
	int GROUP_B_SIZE;
	int GROUP_C_SIZE;
	int NUM_BLOCKS;
};

Settings initSettings(int argc, char** argv);

#endif /* SETTINGS_H_ */
