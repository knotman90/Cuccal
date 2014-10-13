/*
 * memoryLinearizationUtils.cuh
 *
 *  Created on: 09/ott/2014
 *      Author: knotman
 */

#ifndef MEMORYLINEARIZATIONUTILS_CUH_
#define MEMORYLINEARIZATIONUTILS_CUH_
#include <math.h>
#include <stdlib.h>



/*STATIC AND UTILITY CA FUNCTIONS*/
__device__ __host__ static unsigned int inline hd_mod (int m, int n){
	return m >= 0 ? m % n : ( n - abs( m%n ) ) % n;
}

__device__ __host__ inline unsigned   int hd_getLinearIndexToroidal2D(unsigned int i, unsigned int j,unsigned int yDim, unsigned int xDim){
	return (hd_mod(i,yDim)*xDim+hd_mod(j,xDim));
}

__device__ __host__ inline unsigned  int hd_getLinearIndexNormal2D(unsigned int i, unsigned int j, unsigned int yDim,unsigned int xDim){
	return (i*xDim+j);
}


__device__ __host__ inline unsigned   int hd_getLinearIndexToroidal3D(unsigned int i, unsigned int j,unsigned int k, unsigned int yDim, unsigned int xDim,unsigned int zDim){
return (hd_mod(k,zDim)*yDim*xDim)+(hd_mod(i,yDim)*xDim+hd_mod(j,xDim));
}

__device__ __host__ inline unsigned  int hd_getLinearIndexNormal3D(unsigned int i, unsigned int j,unsigned int k, unsigned int yDim, unsigned int xDim,unsigned int zDim){
	return (k*yDim*xDim)+(i*xDim+j);
}


#endif /* MEMORYLINEARIZATIONUTILS_CUH_ */
