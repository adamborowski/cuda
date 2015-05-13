/*
 * Utils.h
 *
 *  Created on: 12-05-2015
 *      Author: adam
 */

#ifndef UTILS_H_
#define UTILS_H_

float* ReadFile(const char* name, int* count);
int divceil(int a, int b);
void testIO();
void initCuda(int argc, char ** argv);
#endif /* UTILS_H_ */
