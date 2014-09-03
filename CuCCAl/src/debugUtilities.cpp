/*
 * debugUtilities.cpp
 *
 *  Created on: 20/mar/2014
 *      Author: davide
 */

#include "enums.h"
#include "debugUtilities.h"

void printDebug(char* msg){
	if(VERBOSE_DEBUG)
		printf("%s\n",msg);
}
