/*
 * enums.h
 *
 *  Created on: 20/mar/2014
 *      Author: davide
 */

#ifndef ENUMS_H_
#define ENUMS_H_
#include <stdio.h>
#include <string.h>
#include <iostream>
using std::string;

#define VERBOSE_DEBUG (1)
// uncomment to disable assert()
// #define NDEBUG

enum TYPE{
	INT,
	CHAR,
	DOUBLE,
	FLOAT,
	BOOL

};

enum NEIGHBORHOOD_TYPE{
	MOORE,
	VON_NEUMANN

};




#endif /* ENUMS_H_ */
