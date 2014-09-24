/*
 * CA.h
 *
 *  Created on: 20/mar/2014
 *      Author: davide
 */

#ifndef CA_H_
#define CA_H_

#include "enums.h"
#include <cassert>
#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <stdbool.h>
#include "config.h"
#include "debugUtilities.h"
#include "IO_Utils.h"
#include "CA_GPU.cuh"
#include <time.h>
#include <vector_types.h>//cuda dim3 include

#define DEFAULT_BLOCKDIM_X (8)
#define DEFAULT_BLOCKDIM_Y (8)

/*STATIC AND UTILITY CA FUNCTIONS*/
__device__ __host__ static unsigned int inline mod (int m, int n){
	return m >= 0 ? m % n : ( n - abs( m%n ) ) % n;
}

 __device__ __host__ inline unsigned   int getLinearIndexToroidal(unsigned int i, unsigned int j,unsigned int rows, unsigned int cols){
	return (mod(i,rows)*cols+mod(j,cols));
}

 __device__ __host__ inline unsigned  int getLinearIndexNormal(unsigned int i, unsigned int j, unsigned int rows,unsigned int cols){
	return (i*cols+j);
}




/*END UTILITY CA FUNCTION*/

struct CA{
	//HAndle to the gpu structure but pointer on GPU
	CA_GPU* d_CA_TOCOPY;
	void** d_subPointer;//handle(CPU) to GPU allocated substates(for cleanup)
	//HANDLE TO THE GPU STRUCTURE!
	CA_GPU* d_CA;
	//launch configuration parameters
	dim3 blockDim;
	dim3 dimGrid;


	//VARIABLES----------------------------------------------
	void(*callback)(unsigned int);
	unsigned int stepsBetweenCallback;
	unsigned long long int steps;
	//flags
	bool isToroidal;
	unsigned int(* getLinearIndex)(unsigned int , unsigned int,unsigned int , unsigned int );
	/*Dimension CA*/
	unsigned int rows;
	unsigned int cols;
	unsigned int numCells;

	/*SUBSTATES list.*/
	void** substates;
	TYPE* substateTypes; //same dimesion of substate_count. substate[i] is of type substateTypes[i].
	unsigned int substates_size; //final size of the substate array
	unsigned int substate_count; //count of the allocated and created substates
	/*END SUBSTATES*/

	/*Transition Function
	 * elementary processes make-up the transition function
	 * They will be executed in order from 0->elementaryProcesses_size*/
	void  (** elementaryProcesses)(CA_GPU* d_CA);
	unsigned int elementaryProcesses_size;
	unsigned int elementaryProcesses_count;

	bool(* stopCondition)();
	/*end Transition Function*/


	bool stop;
	double elapsedTime;

	//METHODS--------------------------------------------------

	//GPU FUNCTIONS
	void initializeGPUAutomata();
	void* allocateGPUBuffer(void * d_buffer,TYPE type);
	void copyBufferToGPU(void* d_to, void*h_to, TYPE type);
	void copyBufferFromGPU(void* h_to, void* d_from, TYPE type);
	void copyBuffersFromGPU();
	void cleanUpGPUAutomata();


	//END GPU FUNCTIONS

	bool checkAutomataStatusBeforeComputation();
	void globalTransitionFunction_MAINLOOP();


	unsigned int getToroidalLinearIndex(unsigned int linearIndex);

	int loadSubstate(SUBSTATE_LABEL substateLabel,const char* const pathToFile);
	int saveSubstate(SUBSTATE_LABEL substateLabel,const char* const pathToFile);

	void printSubstate_STDOUT(SUBSTATE_LABEL substateLabel);
	void printSubstate_STDOUT(SUBSTATE_LABEL substateLabel, unsigned int row, unsigned int col);

	/* START GET SUBSTATE FAMILY FUNCTION*/
	/*2D COORDINATE FUNCTIONS*/
	 bool getSubstateValue_BOOL(unsigned int substate,unsigned int i, unsigned int j) const;
	 double getSubstateValue_DOUBLE(unsigned int substate,unsigned int i, unsigned int j) const;
	 float getSubstateValue_FLOAT(unsigned int substate,unsigned int i, unsigned int j)const;
	 int getSubstateValue_INT(unsigned int substate,unsigned int i, unsigned int j)const;
	 char getSubstateValue_CHAR(unsigned int substate,unsigned int i, unsigned int j)const;

	//-----LINEARIZED COORDINATE FUNCTIONS

	 bool getSubstateValue_BOOL(unsigned int substate,unsigned int index) const;
	 double getSubstateValue_DOUBLE(unsigned int substate,unsigned int index) const;
	 float getSubstateValue_FLOAT(unsigned int substate,unsigned int index)const;
	 int getSubstateValue_INT(unsigned int substate,unsigned int index)const;
	 char getSubstateValue_CHAR(unsigned int substate,unsigned int index)const;


	/* END GET SUBSTATE VALUE FAMILY*/

	/* START SET SUBSTATE FAMILY FUNCTION*/
	 void setSubstateValue_BOOL(unsigned int substate,unsigned int i, unsigned int j,bool const value);
	 void setSubstateValue_DOUBLE(unsigned int substate,unsigned int i, unsigned int j,double const value);
	 void setSubstateValue_FLOAT(unsigned int substate,unsigned int i, unsigned int j, float const value);
	 void setSubstateValue_INT(unsigned int substate,unsigned int i, unsigned int j,int const value);
	 void setSubstateValue_CHAR(unsigned int substate,unsigned int i, unsigned int j,char const value);

	//LINEARIXED COORDINATE FUNCTIONS
	 void setSubstateValue_BOOL(unsigned int substate,unsigned int index,bool const value);
	 void setSubstateValue_DOUBLE(unsigned int substate,unsigned int index,double const value);
	 void setSubstateValue_FLOAT(unsigned int substate,unsigned int index, float const value);
	 void setSubstateValue_INT(unsigned int substate,unsigned int index,int const value);
	 void setSubstateValue_CHAR(unsigned int substate,unsigned int index,char const value);


	/* END set SUBSTATE VALUE FAMILY*/


	unsigned int getNeighborIndex_MOORE(unsigned int i, unsigned int j,unsigned int neighbor);
	unsigned int getNeighborIndex_MOORE(unsigned int index,unsigned int neighbor);


	void registerStopCondictionCallback(bool(*stopCondition)());

	void globalTransitionFunction();

	void registerElementaryProcess( void(*callback)(CA_GPU*) );

	void setInitialParameters(unsigned int substates_size,unsigned int transitionFunction_size);
	void initialize();

	void cleanup();

	void addSubstate(SUBSTATE_LABEL label,TYPE t);

	void registerSubstate(void * buffer,SUBSTATE_LABEL label,TYPE t);

	void* allocateSubstate(TYPE t,void* buffer);


	CA(int rows,int cols,bool toroidal);

	//getter and setter
	unsigned int getCols() const;
	unsigned int getElementaryProcessesSize() const;
	unsigned int getRows() const;
	unsigned int getSubstatesSize() const;
	unsigned long long int getSteps() const;

	unsigned int getBlockdimX() const;

	void setBlockdimX(unsigned int dimX);

	unsigned int getBlockDimY() const;

	void setBlockDimY(unsigned int dimY);

	void updateDimGrid();

	unsigned int isPowerOfTwo (unsigned int x);
	unsigned int getStepsBetweenCopy() const;
	void setStepsBetweenCopy(unsigned int stepsBetweenCopy);
	void setCallback(void(*)(unsigned int));

};




#endif /* CA_H_ */
