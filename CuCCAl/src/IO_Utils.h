/*
 * IO_Utils.h
 *
 *  Created on: 21/mar/2014
 *      Author: Donato D'ambrosio
 *
 *
 */

#ifndef IO_Utils_h
#define IO_Utils_h



#include "CA2D.cuh"
#include <fstream>
using std::fstream;



 enum io_file_state {
	ERROR_OPENING_FILE,
	SUCCESS_OPENING_FILE
};

//save/load -> to/from FILES
template <typename type>
void CA_load_substate_FILE(fstream& f, type* const Q,unsigned int rows,unsigned int cols);

template <typename type>
int CA_load_substate_FILE(const char* const path, type*  const Q,unsigned int rows,unsigned int cols);

template <typename type>
void CA_save_substate_FILE(fstream& f, const type* const Q,unsigned int rows,unsigned int cols);

template <typename type>
int CA_save_substate_FILE(const char* const path, const type* const Q,unsigned int rows,unsigned int cols);

template <typename type>
void CA_print_STDOUT(const type* const Q,unsigned int rows,unsigned int cols);





#endif
