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
#include "CA3D_GPU.cuh"





__device__ unsigned int CA_GPU3D::getNeighborIndex_MOORE_Toroidal(unsigned int i, unsigned int j, unsigned int k,unsigned int neighbor,unsigned int yDim, unsigned int xDim, unsigned int zDim){
	switch(neighbor){

	case 0:
		return hd_getLinearIndexToroidal3D(i,j,k,yDim,xDim,zDim);
	case 1:
		return hd_getLinearIndexToroidal3D(i,j,k,yDim,xDim,zDim);//one row up
	case 2:
		return hd_getLinearIndexToroidal3D(i,j-1,k,yDim,xDim,zDim);//same row one coloumn left
	case 3:
		return hd_getLinearIndexToroidal3D(i,j+1,k,yDim,xDim,zDim);//same row one coloumn right
	case 4:
		return hd_getLinearIndexToroidal3D(i+1,j,k,yDim,xDim,zDim);//same column one row down
	case 5:
		return hd_getLinearIndexToroidal3D(i-1,j-1,k,yDim,xDim,zDim);//one row up one col left
	case 6:
		return hd_getLinearIndexToroidal3D(i+1,j-1,k,yDim,xDim,zDim);//one row down one col left
	case 7:
		return hd_getLinearIndexToroidal3D(i+1,j+1,k,yDim,xDim,zDim);//row down col right
	case 8:
		return hd_getLinearIndexToroidal3D(i-1,j+1,k,yDim,xDim,zDim);//row up col right

		//layer -1 K=k-1
	case 9:
		return hd_getLinearIndexToroidal3D(i,j,k-1,yDim,xDim,zDim);
	case 10:
		return hd_getLinearIndexToroidal3D(i,j,k-1,yDim,xDim,zDim);//one row up
	case 11:
		return hd_getLinearIndexToroidal3D(i,j-1,k-1,yDim,xDim,zDim);//same row one coloumn left
	case 12:
		return hd_getLinearIndexToroidal3D(i,j+1,k-1,yDim,xDim,zDim);//same row one coloumn right
	case 13:
		return hd_getLinearIndexToroidal3D(i+1,j,k-1,yDim,xDim,zDim);//same column one row down
	case 14:
		return hd_getLinearIndexToroidal3D(i-1,j-1,k-1,yDim,xDim,zDim);//one row up one col left
	case 15:
		return hd_getLinearIndexToroidal3D(i+1,j-1,k-1,yDim,xDim,zDim);//one row down one col left
	case 16:
		return hd_getLinearIndexToroidal3D(i+1,j+1,k-1,yDim,xDim,zDim);//row down col right
	case 17:
		return hd_getLinearIndexToroidal3D(i-1,j+1,k-1,yDim,xDim,zDim);//row up col right
		//layer 1 K=k+1
	case 18:
		return hd_getLinearIndexToroidal3D(i,j,k+1,yDim,xDim,zDim);
	case 19:
		return hd_getLinearIndexToroidal3D(i,j,k+1,yDim,xDim,zDim);//one row up
	case 20:
		return hd_getLinearIndexToroidal3D(i,j-1,k+1,yDim,xDim,zDim);//same row one coloumn left
	case 21:
		return hd_getLinearIndexToroidal3D(i,j+1,k+1,yDim,xDim,zDim);//same row one coloumn right
	case 22:
		return hd_getLinearIndexToroidal3D(i+1,j,k+1,yDim,xDim,zDim);//same column one row down
	case 23:
		return hd_getLinearIndexToroidal3D(i-1,j-1,k+1,yDim,xDim,zDim);//one row up one col left
	case 24:
		return hd_getLinearIndexToroidal3D(i+1,j-1,k+1,yDim,xDim,zDim);//one row down one col left
	case 25:
		return hd_getLinearIndexToroidal3D(i+1,j+1,k+1,yDim,xDim,zDim);//row down col right
	case 26:
		return hd_getLinearIndexToroidal3D(i-1,j+1,k+1,yDim,xDim,zDim);//row up col right
	}
	return NULL;

}



/* ------------------START GET SUBSTATE FAMILY FUNCTION------------------*/
__device__   bool CA_GPU3D::getSubstateValue_BOOL(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k) const{
	return ((bool*)d_substates[substateLabel])[hd_getLinearIndexNormal3D(i,j,k,scalars->yDim,scalars->xDim, scalars->zDim)];
}

__device__ double CA_GPU3D::getSubstateValue_DOUBLE(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k)const{

	return ((double*) d_substates[substateLabel])[hd_getLinearIndexNormal3D(i,j,k, scalars->yDim, scalars->xDim, scalars->zDim)];
}

__device__ float CA_GPU3D::getSubstateValue_FLOAT(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k)const{
	return ((float*) d_substates[substateLabel])[hd_getLinearIndexNormal3D(i,j,k, scalars->yDim, scalars->xDim, scalars->zDim)];
}

__device__ int CA_GPU3D::getSubstateValue_INT(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k)const{
	return ((int*) d_substates[substateLabel])[hd_getLinearIndexNormal3D(i,j, k,scalars->yDim, scalars->xDim, scalars->zDim)];
}

__device__ char CA_GPU3D::getSubstateValue_CHAR(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k)const{
	return ((char*) d_substates[substateLabel])[hd_getLinearIndexNormal3D(i,j,k, scalars->yDim, scalars->xDim, scalars->zDim)];
}

//mono index cell representation
__device__ bool CA_GPU3D::getSubstateValue_BOOL(unsigned int substateLabel,unsigned int index) const{
	return ((bool*) d_substates[substateLabel])[index];
}

__device__ double CA_GPU3D::getSubstateValue_DOUBLE(unsigned int substateLabel,unsigned int index)const{
	return ((double*) d_substates[substateLabel])[index];
}

__device__ float CA_GPU3D::getSubstateValue_FLOAT(unsigned int substateLabel,unsigned int index)const{
	return ((float*) d_substates[substateLabel])[index];
}

__device__ int CA_GPU3D::getSubstateValue_INT(unsigned int substateLabel,unsigned int index)const{
	return ((int*) d_substates[substateLabel])[index];
}

__device__ char CA_GPU3D::getSubstateValue_CHAR(unsigned int substateLabel,unsigned int index)const{
	return ((char*) d_substates[substateLabel])[index];
}




/* ------------------END GET SUBSTATE VALUE FAMILY------------------*/


/* ----------------START SET SUBSTATE FAMILY FUNCTION ------------------*/
__device__ void CA_GPU3D::setSubstateValue_BOOL(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k,bool const value) {
	((bool*) d_substates[substateLabel])[hd_getLinearIndexNormal3D(i,j,k, scalars->yDim, scalars->xDim,scalars->zDim)]=value;
}

__device__ void CA_GPU3D::setSubstateValue_DOUBLE(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k, double const value){
	((double*) d_substates[substateLabel])[hd_getLinearIndexNormal3D(i,j,k, scalars->yDim, scalars->xDim,scalars->zDim)]=value;
}

__device__ void CA_GPU3D::setSubstateValue_FLOAT(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k,float const value){
	((float*) d_substates[substateLabel])[hd_getLinearIndexNormal3D(i,j,k, scalars->yDim, scalars->xDim,scalars->zDim)]=value;
}

__device__ void CA_GPU3D::setSubstateValue_INT(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k,int const value){
	((int*) d_substates[substateLabel])[hd_getLinearIndexNormal3D(i,j,k, scalars->yDim, scalars->xDim,scalars->zDim)]=value;
}

__device__ void CA_GPU3D::setSubstateValue_CHAR(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k,char const value){
	((char*) d_substates[substateLabel])[hd_getLinearIndexNormal3D(i,j,k, scalars->yDim, scalars->xDim,scalars->zDim)]=value;
}


__device__ void CA_GPU3D::setSubstateValue_BOOL(unsigned int substateLabel,unsigned int index,bool const value) {
	((bool*) d_substates[substateLabel])[index]=value;
}

__device__ void CA_GPU3D::setSubstateValue_DOUBLE(unsigned int substateLabel,unsigned int index, double const value){
	((double*) d_substates[substateLabel])[index]=value;
}

__device__ void CA_GPU3D::setSubstateValue_FLOAT(unsigned int substateLabel,unsigned int index,float const value){
	((float*) d_substates[substateLabel])[index]=value;
}

__device__ void CA_GPU3D::setSubstateValue_INT(unsigned int substateLabel,unsigned int index,int const value){
	((int*) d_substates[substateLabel])[index]=value;
}

__device__ void CA_GPU3D::setSubstateValue_CHAR(unsigned int substateLabel,unsigned int index,char const value){
	((char*) d_substates[substateLabel])[index]=value;
}


/* ------------------END SET SUBSTATE VALUE FAMILY------------------*/







