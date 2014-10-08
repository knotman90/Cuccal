/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
//#include "CA_GPU.cuh"

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#include "CA2D_GPU.cuh"




/*STATIC AND UTILITY CA FUNCTIONS*/
__device__  inline unsigned int  CA_GPU2D::d_mod (int m, int n){
	return m >= 0 ? m % n : ( n - abs( m%n ) ) % n;
}


__device__  inline   unsigned  int CA_GPU2D::d_getLinearIndexToroidal(unsigned int i, unsigned int j,unsigned int yDim, unsigned int xDim){
	return (this->d_mod(i,yDim)*xDim+this->d_mod(j,xDim));
}

__device__ inline   unsigned  int CA_GPU2D::d_getLinearIndexNormal(unsigned int i, unsigned int j, unsigned int yDim,unsigned int xDim){
	return (i*xDim+j);
}

__device__  inline unsigned   int CA_GPU2D::d_getLinearIndexToroidalLinear(unsigned int index,unsigned int yDim, unsigned int xDim){
	return d_mod(index,yDim*xDim);
}


__device__ unsigned int CA_GPU2D::getNeighborIndex_MOORE_Toroidal(unsigned int i, unsigned int j,unsigned int neighbor,unsigned int yDim, unsigned int xDim){
	switch(neighbor){
	case 0:
		return d_getLinearIndexToroidal(i,j,yDim,xDim);
	case 1:
		return d_getLinearIndexToroidal(i-1,j,yDim,xDim);//one row up
	case 2:
		return d_getLinearIndexToroidal(i,j-1,yDim,xDim);//same row one coloumn left
	case 3:
		return d_getLinearIndexToroidal(i,j+1,yDim,xDim);//same row one coloumn right
	case 4:
		return d_getLinearIndexToroidal(i+1,j,yDim,xDim);//same column one row down
	case 5:
		return d_getLinearIndexToroidal(i-1,j-1,yDim,xDim);//one row up one col left
	case 6:
		return d_getLinearIndexToroidal(i+1,j-1,yDim,xDim);//one row down one col left
	case 7:
		return d_getLinearIndexToroidal(i+1,j+1,yDim,xDim);//row down col right
	case 8:
		return d_getLinearIndexToroidal(i-1,j+1,yDim,xDim);//row up col right
	}

	return NULL;//it should never be executed
}








/* ------------------START GET SUBSTATE FAMILY FUNCTION------------------*/
__device__  bool CA_GPU2D::getSubstateValue_BOOL(unsigned int substateLabel,unsigned int i, unsigned int j) const{
	return ((bool*)d_substates[substateLabel])[getLinearIndex(i,j,scalars->yDim,scalars->xDim)];
}

__device__ double CA_GPU2D::getSubstateValue_DOUBLE(unsigned int substateLabel,unsigned int i, unsigned int j)const{

	return ((double*) d_substates[substateLabel])[getLinearIndex(i,j, scalars->yDim, scalars->xDim)];
}

__device__ float CA_GPU2D::getSubstateValue_FLOAT(unsigned int substateLabel,unsigned int i, unsigned int j)const{
	return ((float*) d_substates[substateLabel])[getLinearIndex(i,j, scalars->yDim, scalars->xDim)];
}

__device__ int CA_GPU2D::getSubstateValue_INT(unsigned int substateLabel,unsigned int i, unsigned int j)const{
	return ((int*) d_substates[substateLabel])[this->getLinearIndex(i,j, scalars->yDim, scalars->xDim)];
}

__device__ char CA_GPU2D::getSubstateValue_CHAR(unsigned int substateLabel,unsigned int i, unsigned int j)const{
	return ((char*) d_substates[substateLabel])[getLinearIndex(i,j, scalars->yDim, scalars->xDim)];
}

//mono index cell representation
__device__ bool CA_GPU2D::getSubstateValue_BOOL(unsigned int substateLabel,unsigned int index) const{
	return ((bool*) d_substates[substateLabel])[index];
}

__device__ double CA_GPU2D::getSubstateValue_DOUBLE(unsigned int substateLabel,unsigned int index)const{
	return ((double*) d_substates[substateLabel])[index];
}

__device__ float CA_GPU2D::getSubstateValue_FLOAT(unsigned int substateLabel,unsigned int index)const{
	return ((float*) d_substates[substateLabel])[index];
}

__device__ int CA_GPU2D::getSubstateValue_INT(unsigned int substateLabel,unsigned int index)const{
	return ((int*) d_substates[substateLabel])[index];
}

__device__ char CA_GPU2D::getSubstateValue_CHAR(unsigned int substateLabel,unsigned int index)const{
	return ((char*) d_substates[substateLabel])[index];
}




/* ------------------END GET SUBSTATE VALUE FAMILY------------------*/


/* ----------------START SET SUBSTATE FAMILY FUNCTION ------------------*/
__device__ void CA_GPU2D::setSubstateValue_BOOL(unsigned int substateLabel,unsigned int i, unsigned int j,bool const value) {
	((bool*) d_substates[substateLabel])[getLinearIndex(i,j, scalars->yDim, scalars->xDim)]=value;
}

__device__ void CA_GPU2D::setSubstateValue_DOUBLE(unsigned int substateLabel,unsigned int i, unsigned int j, double const value){
	((double*) d_substates[substateLabel])[getLinearIndex(i,j, scalars->yDim, scalars->xDim)]=value;
}

__device__ void CA_GPU2D::setSubstateValue_FLOAT(unsigned int substateLabel,unsigned int i, unsigned int j,float const value){
	((float*) d_substates[substateLabel])[getLinearIndex(i,j, scalars->yDim, scalars->xDim)]=value;
}

__device__ void CA_GPU2D::setSubstateValue_INT(unsigned int substateLabel,unsigned int i, unsigned int j,int const value){
	((int*) d_substates[substateLabel])[getLinearIndex(i,j, scalars->yDim, scalars->xDim)]=value;
}

__device__ void CA_GPU2D::setSubstateValue_CHAR(unsigned int substateLabel,unsigned int i, unsigned int j,char const value){
	((char*) d_substates[substateLabel])[getLinearIndex(i,j, scalars->yDim, scalars->xDim)]=value;
}


__device__ void CA_GPU2D::setSubstateValue_BOOL(unsigned int substateLabel,unsigned int index,bool const value) {
	((bool*) d_substates[substateLabel])[index]=value;
}

__device__ void CA_GPU2D::setSubstateValue_DOUBLE(unsigned int substateLabel,unsigned int index, double const value){
	((double*) d_substates[substateLabel])[index]=value;
}

__device__ void CA_GPU2D::setSubstateValue_FLOAT(unsigned int substateLabel,unsigned int index,float const value){
	((float*) d_substates[substateLabel])[index]=value;
}

__device__ void CA_GPU2D::setSubstateValue_INT(unsigned int substateLabel,unsigned int index,int const value){
	((int*) d_substates[substateLabel])[index]=value;
}

__device__ void CA_GPU2D::setSubstateValue_CHAR(unsigned int substateLabel,unsigned int index,char const value){
	((char*) d_substates[substateLabel])[index]=value;
}


/* ------------------END SET SUBSTATE VALUE FAMILY------------------*/



__device__ unsigned int CA_GPU2D::getLinearIndex(unsigned int i, unsigned int j,unsigned int yDim, unsigned int xDim)const{
	return (i*xDim+j);
}



