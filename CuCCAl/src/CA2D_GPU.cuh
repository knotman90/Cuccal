/*
 * CA_GPU.cuh
 *
 *  Created on: 27/mar/2014
 *      Author: davide
 */



#ifndef CA_GPU_CUH_
#define CA_GPU_CUH_

#include "enums.h"
#include <vector_types.h>//cuda dim3 include

struct CA2D;


struct SCALARS_CA_GPU2D{
	/*Dimension CA*/
	unsigned int yDim;
	unsigned int xDim;
	unsigned int numCells;

	unsigned long long int steps;
	bool isToroidal;

	unsigned int substates_size; //final size of the substate array

	unsigned int elementaryProcesses_size;
	bool stop;
};

struct CA_GPU2D{
	//VARIABLES----------------------------------------------

	SCALARS_CA_GPU2D* scalars;


	//flags

	// unsigned int(* getLinearIndex)(unsigned int , unsigned int,unsigned int , unsigned int );


	/*SUBSTATES list.*/

	void** d_substates;
	TYPE* d_substateTypes; //same dimesion of substate_count. substate[i] is of type substateTypes[i].

	/*END SUBSTATES*/

	/*Transition Function
	 * elementary processes make-up the transition function
	 * They will be executed in order from 0->elementaryProcesses_size*/
	void (** d_elementaryProcesses)(unsigned  int, unsigned  int);


	bool(* stopCondition)();
	/*end Transition Function*/



	//METHODS--------------------------------------------------
	/* START GET SUBSTATE FAMILY FUNCTION*/
	/*2D COORDINATE FUNCTIONS*/
	__device__  bool getSubstateValue_BOOL(unsigned int substate,unsigned int i, unsigned int j) const;
	__device__ double getSubstateValue_DOUBLE(unsigned int substate,unsigned int i, unsigned int j) const;
	__device__ float getSubstateValue_FLOAT(unsigned int substate,unsigned int i, unsigned int j)const;
	__device__ int getSubstateValue_INT(unsigned int substate,unsigned int i, unsigned int j)const;
	__device__  char getSubstateValue_CHAR(unsigned int substate,unsigned int i, unsigned int j)const;

	//-----LINEARIZED COORDINATE FUNCTIONS

	__device__ bool getSubstateValue_BOOL(unsigned int substate,unsigned int index) const;
	__device__ double getSubstateValue_DOUBLE(unsigned int substate,unsigned int index) const;
	__device__ float getSubstateValue_FLOAT(unsigned int substate,unsigned int index)const;
	__device__ int getSubstateValue_INT(unsigned int substate,unsigned int index)const;
	__device__ char getSubstateValue_CHAR(unsigned int substate,unsigned int index)const;


	/* END GET SUBSTATE VALUE FAMILY*/

	/* START SET SUBSTATE FAMILY FUNCTION*/
	__device__ void setSubstateValue_BOOL(unsigned int substate,unsigned int i, unsigned int j,bool const value);
	__device__ void setSubstateValue_DOUBLE(unsigned int substate,unsigned int i, unsigned int j,double const value);
	__device__ void setSubstateValue_FLOAT(unsigned int substate,unsigned int i, unsigned int j, float const value);
	__device__ void setSubstateValue_INT(unsigned int substate,unsigned int i, unsigned int j,int const value);
	__device__ void setSubstateValue_CHAR(unsigned int substate,unsigned int i, unsigned int j,char const value);

	//LINEARIXED COORDINATE FUNCTIONS
	__device__ void setSubstateValue_BOOL(unsigned int substate,unsigned int index,bool const value);
	__device__ void setSubstateValue_DOUBLE(unsigned int substate,unsigned int index,double const value);
	__device__ void setSubstateValue_FLOAT(unsigned int substate,unsigned int index, float const value);
	__device__ void setSubstateValue_INT(unsigned int substate,unsigned int index,int const value);
	__device__ void setSubstateValue_CHAR(unsigned int substate,unsigned int index,char const value);

	__device__ unsigned int getLinearIndex(unsigned int , unsigned int ,unsigned int , unsigned int )const;

	//utility functions
	__device__  inline  unsigned int  d_mod (int m, int n);
	__device__  inline   unsigned  int d_getLinearIndexToroidal(unsigned int i, unsigned int j,unsigned int rows, unsigned int cols);
	__device__   inline  unsigned  int d_getLinearIndexNormal(unsigned int i, unsigned int j, unsigned int rows,unsigned int cols);

	__device__  inline   unsigned  int d_getLinearIndexToroidalLinear(unsigned int index,unsigned int rows, unsigned int cols);


	__device__ unsigned int getNeighborIndex_MOORE_Toroidal(unsigned int i, unsigned int j,unsigned int neighbor,unsigned int rows, unsigned int cols);
	__device__ unsigned int getNeighborIndex_MOORE_Toroidal(unsigned int index,unsigned int neighbor);


};

#endif /* CA_GPU_CUH_ */
