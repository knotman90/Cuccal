/*
 * CA_GPU.cuh
 *
 *  Created on: 27/mar/2014
 *      Author: davide
 */

#ifndef CA_GPU3D_CUH_
#define CA_GPU3D_CUH_

#include "../enums.h"
#include <vector_types.h>//cuda dim3 include

struct CA3D;


struct SCALARS_CA_GPU3D{
	/*Dimension CA*/
	unsigned int yDim;
	unsigned int xDim;
	unsigned int zDim;
	unsigned int numCells;

	unsigned long long int steps;
	bool isToroidal;

	unsigned int substates_size; //final size of the substate array

	unsigned int elementaryProcesses_size;
	bool stop;
};

struct CA_GPU3D{
	//VARIABLES----------------------------------------------

	SCALARS_CA_GPU3D* scalars;

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
	__device__  bool getSubstateValue_BOOL(unsigned int substate,unsigned int i, unsigned int j, unsigned int k) const;
	__device__ double getSubstateValue_DOUBLE(unsigned int substate,unsigned int i, unsigned int j, unsigned int k) const;
	__device__ float getSubstateValue_FLOAT(unsigned int substate,unsigned int i, unsigned int j, unsigned int k)const;
	__device__ int getSubstateValue_INT(unsigned int substate,unsigned int i, unsigned int j, unsigned int k)const;
	__device__  char getSubstateValue_CHAR(unsigned int substate,unsigned int i, unsigned int j, unsigned int k)const;

	//-----LINEARIZED COORDINATE FUNCTIONS

	__device__ bool getSubstateValue_BOOL(unsigned int substate,unsigned int index) const;
	__device__ double getSubstateValue_DOUBLE(unsigned int substate,unsigned int index) const;
	__device__ float getSubstateValue_FLOAT(unsigned int substate,unsigned int index)const;
	__device__ int getSubstateValue_INT(unsigned int substate,unsigned int index)const;
	__device__ char getSubstateValue_CHAR(unsigned int substate,unsigned int index)const;


	/* END GET SUBSTATE VALUE FAMILY*/

	/* START SET SUBSTATE FAMILY FUNCTION*/
	__device__ void setSubstateValue_BOOL(unsigned int substate,unsigned int i, unsigned int j,unsigned int k,bool const value);
	__device__ void setSubstateValue_DOUBLE(unsigned int substate,unsigned int i, unsigned int j,unsigned int k,double const value);
	__device__ void setSubstateValue_FLOAT(unsigned int substate,unsigned int i, unsigned int j,unsigned int k, float const value);
	__device__ void setSubstateValue_INT(unsigned int substate,unsigned int i, unsigned int j,unsigned int k,int const value);
	__device__ void setSubstateValue_CHAR(unsigned int substate,unsigned int i, unsigned int j,unsigned int k,char const value);

	//LINEARIZED COORDINATE FUNCTIONS
	__device__ void setSubstateValue_BOOL(unsigned int substate,unsigned int index,bool const value);
	__device__ void setSubstateValue_DOUBLE(unsigned int substate,unsigned int index,double const value);
	__device__ void setSubstateValue_FLOAT(unsigned int substate,unsigned int index, float const value);
	__device__ void setSubstateValue_INT(unsigned int substate,unsigned int index,int const value);
	__device__ void setSubstateValue_CHAR(unsigned int substate,unsigned int index,char const value);

	__device__ unsigned int getLinearIndex(unsigned int i, unsigned int j,unsigned int k,unsigned int yDim, unsigned int xDim, unsigned int zDim)const;

	//utility functions
	__device__  inline  unsigned int  d_mod (int m, int n);
	__device__  inline   unsigned  int d_getLinearIndexToroidal(unsigned int i, unsigned int j,unsigned int k,unsigned int yDim, unsigned int xDim,unsigned int zDim);
	__device__   inline  unsigned  int d_getLinearIndexNormal(unsigned int i, unsigned int j,unsigned int k, unsigned int yDim,unsigned int xDim,unsigned int zDim);

	__device__  inline   unsigned  int d_getLinearIndexToroidalLinear(unsigned int index,unsigned int yDim, unsigned int xDim,unsigned int zDim);


	__device__ unsigned int getNeighborIndex_MOORE_Toroidal(unsigned int i, unsigned int j, unsigned int k,unsigned int neighbor,unsigned int yDim, unsigned int xDim, unsigned int zDim);
	__device__ unsigned int getNeighborIndex_MOORE_Toroidal(unsigned int index,unsigned int neighbor);


};

#endif /* CA_GPU_CUH_ */
