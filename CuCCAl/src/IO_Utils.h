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



#include "CA2D/CA2D.cuh"
#include <fstream>
using std::fstream;



 enum io_file_state {
	ERROR_OPENING_FILE,
	SUCCESS_OPENING_FILE
};
//2D IO-functions
//save/load -> to/from FILES
template <typename type>
void CA_load_substate_FILE2D(fstream& f, type* const Q,unsigned int yDim,unsigned int xDim);

template <typename type>
int CA_load_substate_FILE2D(const char* const path, type*  const Q,unsigned int yDim,unsigned int xDim);

template <typename type>
void CA_save_substate_FILE2D(fstream& f, const type* const Q,unsigned int yDim,unsigned int xDim);

template <typename type>
int CA_save_substate_FILE2D(const char* const path, const type* const Q,unsigned int yDim,unsigned int xDim);

template <typename type>
void CA_print_STDOUT2D(const type* const Q,unsigned int yDim,unsigned int xDim);



//save/load -> to/from FILES
template <typename type>
void CA_load_substate_FILE3D(fstream& f, type* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);

template <typename type>
int CA_load_substate_FILE3D(const char* const path, type*  const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);

template <typename type>
void CA_save_substate_FILE3D(fstream& f, const type* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);

template <typename type>
int CA_save_substate_FILE3D(const char* const path, const type* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);

template <typename type>
void CA_print_STDOUT3D(const type* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);






#endif
