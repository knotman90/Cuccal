/*
 * config.h
 *
 *  Created on: 20/mar/2014
 *      Author: davide
 */

/*
         5 | 1 | 8
        ---|---|---
         2 | 0 | 3
        ---|---|---
         6 | 4 | 7
 */

#include "config.h"
#include "CA.cuh"



//CA dichiarata in CA.h
extern CA CA;



__global__ void gpuEvolve(CA_GPU* d_CA){
	unsigned int col=(threadIdx.x+blockIdx.x*blockDim.x);
	unsigned int row=(threadIdx.y+blockIdx.y*blockDim.y);
	unsigned int totRows=d_CA->scalars->rows;
	unsigned int totCols=d_CA->scalars->cols;
	if(row<totRows && col<totCols){
		short unsigned int count=0;
		unsigned int linNeighIdx=0;
		bool alive=d_CA->getSubstateValue_BOOL(Q,row,col);
		for (int neigh = 1; neigh < 9; neigh++) {
			linNeighIdx=d_CA->getNeighborIndex_MOORE_Toroidal(row,col,neigh,totRows,totCols);
			if(d_CA->getSubstateValue_BOOL(Q,linNeighIdx)==true){
				count++;
			}
		}
		alive=((!alive && count==3) || (alive && ( count==2 || count==3))) ? true : false;
		d_CA->setSubstateValue_BOOL(Q_NEW,row,col,alive);
	}

}


void __global__ copyBoard(CA_GPU* d_CA){
	int col=(threadIdx.x+blockIdx.x*blockDim.x);
	int row=(threadIdx.y+blockIdx.y*blockDim.y);
	if(row<d_CA->scalars->rows && col<d_CA->scalars->cols){
		d_CA->setSubstateValue_BOOL(Q,row,col,d_CA->getSubstateValue_BOOL(Q_NEW,row,col));
	}

}




//true means --> STOP THE AUTOMATA
bool stopCondition(){

	if(CA.getSteps()>100){
		return true;
	}
	return false;
}


