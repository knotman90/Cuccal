/*
 * CA.h
 *
 *  Created on: 20/mar/2014
 *      Author: davide
 */

#ifndef CA_H_
#define CA_H_

#include "enums.h"
#include "config.h"
#include "debugUtilities.h"
#include "IO_Utils.h"
#include "CA2D_GPU.cuh"
#include "memoryLinearizationUtils.cuh"//mod 2/3D to 1D memory layout

#include <cassert>
#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <stdbool.h>
#include <time.h>
#include <vector_types.h>//cuda dim3 include

#define DEFAULT_BLOCKDIM_X (8)
#define DEFAULT_BLOCKDIM_Y (8)
#define DEFAULT_BLOCKDIM_Z (8)







struct CA2D{
	//HAndle to the gpu structure but pointer on GPU
	CA_GPU2D* d_CA_TOCOPY;
	void** d_subPointer;//handle(CPU) to GPU allocated substates(for cleanup)
	//HANDLE TO THE GPU STRUCTURE!
	CA_GPU2D* d_CA;
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
	unsigned int xDim;
	unsigned int yDim;




	unsigned int numCells;//2D=(yDim*xDim) 3D=(x*y*z);

	/*SUBSTATES list.*/
	void** substates;
	TYPE* substateTypes; //same dimesion of substate_count. substate[i] is of type substateTypes[i].
	unsigned int substates_size; //final size of the substate array
	unsigned int substate_count; //count of the allocated and created substates
	/*END SUBSTATES*/

	/*Transition Function
	 * elementary processes make-up the transition function
	 * They will be executed in order from 0->elementaryProcesses_size*/
	void  (** elementaryProcesses)(CA_GPU2D* d_CA);
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
	void globalTransitionFunction_MAINLOOP_callback();

	bool evolveOneStep();
	//k > 0
	bool evolveKsteps(unsigned int k);


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
	void setSubstateValue2D_BOOL(unsigned int substate,unsigned int i, unsigned int j,bool const value);
	void setSubstateValue2D_DOUBLE(unsigned int substate,unsigned int i, unsigned int j,double const value);
	void setSubstateValue2D_FLOAT(unsigned int substate,unsigned int i, unsigned int j, float const value);
	void setSubstateValue2D_INT(unsigned int substate,unsigned int i, unsigned int j,int const value);
	void setSubstateValue2D_CHAR(unsigned int substate,unsigned int i, unsigned int j,char const value);

	//LINEARIXED COORDINATE FUNCTIONS
	void setSubstateValue_BOOL(unsigned int substate,unsigned int index,bool const value);
	void setSubstateValue_DOUBLE(unsigned int substate,unsigned int index,double const value);
	void setSubstateValue_FLOAT(unsigned int substate,unsigned int index, float const value);
	void setSubstateValue_INT(unsigned int substate,unsigned int index,int const value);
	void setSubstateValue_CHAR(unsigned int substate,unsigned int index,char const value);


	/* END set SUBSTATE VALUE FAMILY*/


	unsigned int getNeighborIndex2D_MOORE(unsigned int i, unsigned int j,unsigned int neighbor);
	unsigned int getNeighborIndex2D_MOORE(unsigned int index,unsigned int neighbor);




	void registerStopCondictionCallback(bool(*stopCondition)());

	void globalTransitionFunction();

	void registerElementaryProcess( void(*callback)(CA_GPU2D*) );

	void setInitialParameters(unsigned int substates_size,unsigned int transitionFunction_size);
	void initialize();

	void cleanup();

	void addSubstate(SUBSTATE_LABEL label,TYPE t);

	void registerSubstate(void * buffer,SUBSTATE_LABEL label,TYPE t);

	void* allocateSubstate(TYPE t,void* buffer);


	//2D constructor
	CA2D(int XDim,int YDim,bool toroidal);
	void preliminaryCAConstructor();

	CA2D(int XDim,int YDim,int ZDim,bool toroidal);

	//getter and setter
	unsigned int get_xDim() const;
	unsigned int get_yDim() const;
	unsigned int get_zDim() const;

	unsigned int getElementaryProcessesSize() const;
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
