/*
 * CA.cpp
 *
 *  Created on: 21/mar/2014
 *      Author: davide
 */

#include "CA3D.cuh"

#define CUDA_CHECK_RETURN(value) {											\
		cudaError_t _m_cudaStat = value;										\
		if (_m_cudaStat != cudaSuccess) {										\
			fprintf(stderr, "Error %s at line %d in file %s\n",					\
					cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
					exit(1);															\
		} }

//numcells=yDim*xDim in constructor CA
void* CA3D::allocateGPUBuffer(void * d_buffer,TYPE type){
	switch(type){

	case FLOAT:
		CUDA_CHECK_RETURN(cudaMalloc((void**)&d_buffer,sizeof(float)*numCells));
		break;
	case DOUBLE:
		CUDA_CHECK_RETURN(cudaMalloc((void**)&d_buffer,sizeof(double)*numCells));
		break;
	case CHAR:
		CUDA_CHECK_RETURN(cudaMalloc((void**)&d_buffer,sizeof(char)*numCells));
		break;
	case INT:
		CUDA_CHECK_RETURN(cudaMalloc((void**)&d_buffer,sizeof(int)*numCells));
		break;
	case BOOL:
		CUDA_CHECK_RETURN(cudaMalloc((void**)&d_buffer,sizeof(bool)*numCells));
		break;

	}

	return d_buffer;
}


void CA3D::copyBufferFromGPU(void* h_to, void* d_from, TYPE type){

	switch(type){

	case FLOAT:
		CUDA_CHECK_RETURN(cudaMemcpy(h_to,d_from,sizeof(float)*numCells,cudaMemcpyDeviceToHost));
		break;
	case DOUBLE:
		CUDA_CHECK_RETURN(cudaMemcpy(h_to,d_from,sizeof(double)*numCells,cudaMemcpyDeviceToHost));
		break;
	case CHAR:
		CUDA_CHECK_RETURN(cudaMemcpy(h_to,d_from,sizeof(char)*numCells,cudaMemcpyDeviceToHost));
		break;
	case INT:
		CUDA_CHECK_RETURN(cudaMemcpy(h_to,d_from,sizeof(int)*numCells,cudaMemcpyDeviceToHost));
		break;
	case BOOL:
		CUDA_CHECK_RETURN(cudaMemcpy(h_to,d_from,sizeof(bool)*numCells,cudaMemcpyDeviceToHost));
		break;

	}
}

void CA3D::copyBufferToGPU(void* d_to, void* h_from, TYPE type){

	switch(type){

	case FLOAT:
		CUDA_CHECK_RETURN(cudaMemcpy(d_to,h_from,sizeof(float)*numCells,cudaMemcpyHostToDevice));
		break;
	case DOUBLE:
		CUDA_CHECK_RETURN(cudaMemcpy(d_to,h_from,sizeof(double)*numCells,cudaMemcpyHostToDevice));
		break;
	case CHAR:
		CUDA_CHECK_RETURN(cudaMemcpy(d_to,h_from,sizeof(char)*numCells,cudaMemcpyHostToDevice));
		break;
	case INT:
		CUDA_CHECK_RETURN(cudaMemcpy(d_to,h_from,sizeof(int)*numCells,cudaMemcpyHostToDevice));
		break;
	case BOOL:
		CUDA_CHECK_RETURN(cudaMemcpy(d_to,h_from,sizeof(bool)*numCells,cudaMemcpyHostToDevice));
		break;

	}
}

__global__ void initializeDCA(void** d_AllocatedpointerSubstates,TYPE* d_substateTypes,SCALARS_CA_GPU3D* scalarsTOCPY,CA_GPU3D* d_CA){

	d_CA->d_substates=d_AllocatedpointerSubstates;
	d_CA->d_substateTypes=d_substateTypes;
	d_CA->scalars=scalarsTOCPY;


}

__global__ void printValues(CA_GPU3D* d_CA){

	//printf("(%i,%i),(%i,%i)\n",((int*)d_CA->d_substates[0])[threadIdx.x],d_CA->d_substateTypes[0],((int*)d_CA->d_substates[1])[threadIdx.x],d_CA->d_substateTypes[1]);
	//d_CA->d_substates=d_AllocatedpointerSubstates;
	//printf("SCALARS\n yDim=%i, xDim=%i value=%i",d_CA->scalars->yDim,d_CA->scalars->xDim,d_CA->getSubstateValue_INT(Q,threadIdx.x));


}

void CA3D::initializeGPUAutomata(){
	//allocate GPU_CA on GPU
	CUDA_CHECK_RETURN(cudaMalloc(&d_CA,sizeof(CA_GPU3D)));
	d_CA_TOCOPY= new CA_GPU3D();

	//cancellaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
	//		for(int i=0;i<yDim;i++){
	//			for(int j=0;j<xDim;j++){
	//				if(i%substates[Q]==0){
	//					((bool*)substates[Q])[getLinearIndexNormal(i,j,yDim,xDim)]=true;
	//				((bool*)substates[Q_NEW])[getLinearIndexNormal(i,j,yDim,xDim)]=false;
	//			}
	//		}
	//		}
	//glider
	//	((bool*)substates[Q])[getLinearIndexNormal(5,5,yDim,xDim)]=true;
	//	((bool*)substates[Q])[getLinearIndexNormal(6,5,yDim,xDim)]=true;
	//	((bool*)substates[Q])[getLinearIndexNormal(5,6,yDim,xDim)]=true;
	//	((bool*)substates[Q])[getLinearIndexNormal(6,6,yDim,xDim)]=true;
	//
	//	((bool*)substates[Q])[getLinearIndexNormal(7,7,yDim,xDim)]=true;
	//	((bool*)substates[Q])[getLinearIndexNormal(8,7,yDim,xDim)]=true;
	//	((bool*)substates[Q])[getLinearIndexNormal(7,8,yDim,xDim)]=true;
	//	((bool*)substates[Q])[getLinearIndexNormal(8,8,yDim,xDim)]=true;

	//allocate memory ON GPU

	/*allocate all the substates ON GPU
	substate_size=substate_count=real number of registered buffers -> coherent state of the automata
	checked befor of GPU initialization
	conversion between unsigned int(substateTypes) and TYPE is legal*/
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_CA_TOCOPY->d_substates,sizeof(void*)*substates_size));

	d_subPointer = (void**)malloc(sizeof(void*)*substates_size);
	for(int i=0;i<substates_size;i++){
		d_subPointer[i]=allocateGPUBuffer(d_subPointer[i],(TYPE)substateTypes[i]);
		copyBufferToGPU(d_subPointer[i],substates[i],(TYPE)substateTypes[i]);

	}
	CUDA_CHECK_RETURN(cudaMemcpy(d_CA_TOCOPY->d_substates,d_subPointer,sizeof(void*)*substates_size,cudaMemcpyHostToDevice));
	//CUDA_CHECK_RETURN(cudaFree((void*)(&d_CA_TOCOPY->d_substates[1])));

	//substates type array (allocation and copy, that's a constant array usually)
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_CA_TOCOPY->d_substateTypes,sizeof(TYPE)*substates_size));
	CUDA_CHECK_RETURN(cudaMemcpy(d_CA_TOCOPY->d_substateTypes,substateTypes,sizeof(TYPE)*substates_size,cudaMemcpyHostToDevice));

	//copyScalars. First create structure to be copied, then allocate memory on GPU->copy structure on GPU->
	//->then link d_CA_TOCOPY->scalars to d_CA->scalars whithin a kernel
	SCALARS_CA_GPU3D* scalars_TOPCOPY = new SCALARS_CA_GPU3D();
	scalars_TOPCOPY->xDim=xDim;
	scalars_TOPCOPY->yDim=yDim;
	scalars_TOPCOPY->stop=stop;
	scalars_TOPCOPY->steps=steps;
	scalars_TOPCOPY->isToroidal=isToroidal;
	scalars_TOPCOPY->substates_size=substates_size;
	scalars_TOPCOPY->numCells=numCells;

	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_CA_TOCOPY->scalars,sizeof(SCALARS_CA_GPU3D)));
	CUDA_CHECK_RETURN(cudaMemcpy(d_CA_TOCOPY->scalars,scalars_TOPCOPY,sizeof(SCALARS_CA_GPU3D),cudaMemcpyHostToDevice));

	free(scalars_TOPCOPY);//not needed anymore

	cudaThreadSynchronize();
	initializeDCA<<<1,1>>>(d_CA_TOCOPY->d_substates,d_CA_TOCOPY->d_substateTypes,d_CA_TOCOPY->scalars,d_CA);


	cudaThreadSynchronize();
	printValues<<<1,10>>>(d_CA);
	cudaThreadSynchronize();



	printDebug("GPU memory allocated");
}

void CA3D::cleanUpGPUAutomata(){
	//FREE MEMORY ON GPU-> remember to free first all the buffers INSIDE the struct
	printDebug("inizio cleanUP GPU");
	for(int i=0;i<substates_size;i++){
		CUDA_CHECK_RETURN(cudaFree((void*)((d_subPointer[i]))));

	}
	//free scalars GPU
	CUDA_CHECK_RETURN(cudaFree((void*)d_CA_TOCOPY->scalars));
	//CUDA_CHECK_RETURN(cudaFree(d_CA));
	free(d_subPointer);
	printDebug("GPU memory freeed");
}

unsigned long long int CA3D::getSteps() const{
	return steps;
}

unsigned int CA3D::getToroidalLinearIndex(unsigned int linearIndex){
	return mod(linearIndex,yDim*xDim);
}

int CA3D::loadSubstate(SUBSTATE_LABEL substateLabel, const char* const pathToFile){
	short int status =SUCCESS_OPENING_FILE;
	unsigned int type= substateTypes[substateLabel];
	switch(type){
	case FLOAT:
		status=CA_load_substate_FILE(pathToFile,(float*)(substates[substateLabel]),yDim,xDim);
		break;
	case DOUBLE:
		status=CA_load_substate_FILE(pathToFile,(double*)(substates[substateLabel]),yDim,xDim);
		break;
	case CHAR:
		status=CA_load_substate_FILE(pathToFile,(char*)(substates[substateLabel]),yDim,xDim);
		break;
	case INT:
		status=CA_load_substate_FILE(pathToFile,(int*)(substates[substateLabel]),yDim,xDim);
		break;
	case BOOL:
		status=CA_load_substate_FILE(pathToFile,(bool*)(substates[substateLabel]),yDim,xDim);
		break;
	}
	return status;
}

int CA3D::saveSubstate(SUBSTATE_LABEL substateLabel, const char* const pathToFile){
	short int status =SUCCESS_OPENING_FILE;
	unsigned int type= substateTypes[substateLabel];
	switch(type){
	case FLOAT:
		status=CA_save_substate_FILE(pathToFile,(float*)(substates[substateLabel]),yDim,xDim);
		break;
	case DOUBLE:
		status=CA_save_substate_FILE(pathToFile,(double*)(substates[substateLabel]),yDim,xDim);
		break;
	case CHAR:
		status=CA_save_substate_FILE(pathToFile,(char*)(substates[substateLabel]),yDim,xDim);
		break;
	case INT:
		status=CA_save_substate_FILE(pathToFile,(int*)(substates[substateLabel]),yDim,xDim);
		break;
	case BOOL:
		status=CA_save_substate_FILE(pathToFile,(bool*)(substates[substateLabel]),yDim,xDim);
		break;

	}
	return status;
}



void CA3D::printSubstate_STDOUT(SUBSTATE_LABEL substateLabel){
	printSubstate_STDOUT(substateLabel,yDim,xDim);

}

void CA3D::printSubstate_STDOUT(SUBSTATE_LABEL substateLabel, unsigned int Nrow, unsigned int Ncol){
	assert(Nrow<=yDim && Ncol<=xDim );

	unsigned int type= substateTypes[substateLabel];
	switch(type){
	case FLOAT:
		CA_print_STDOUT((float*)(substates[substateLabel]),Nrow,Ncol);
		break;
	case DOUBLE:
		CA_print_STDOUT((double*)(substates[substateLabel]),Nrow,Ncol);
		break;
	case CHAR:
		CA_print_STDOUT((char*)(substates[substateLabel]),Nrow,Ncol);
		break;
	case INT:
		CA_print_STDOUT((int*)(substates[substateLabel]),Nrow,Ncol);
		break;
	case BOOL:
		CA_print_STDOUT((bool*)(substates[substateLabel]),Nrow,Ncol);
		break;

	}

}


/* ------------------START GET SUBSTATE FAMILY FUNCTION------------------*/
bool CA3D::getSubstateValue_BOOL(unsigned int substateLabel,unsigned int i, unsigned int j, unsigned int k) const{
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==BOOL);
	return ((bool*)substates[substateLabel])[getLinearIndex(i,j,k,yDim,xDim,zDim)];
}

double CA3D::getSubstateValue_DOUBLE(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k)const{
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==DOUBLE);
	return ((double*)substates[substateLabel])[getLinearIndex(i,j,k,yDim,xDim,zDim)];
}

float CA3D::getSubstateValue_FLOAT(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k)const{
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==FLOAT);
	return ((float*)substates[substateLabel])[getLinearIndex(i,j,k,yDim,xDim,zDim)];
}

int CA3D::getSubstateValue_INT(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k)const{
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==INT);
	return ((int*)substates[substateLabel])[getLinearIndex(i,j,k,yDim,xDim,zDim)];
}

char CA3D::getSubstateValue_CHAR(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k)const{
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==CHAR);
	return ((char*)substates[substateLabel])[getLinearIndex(i,j,k,yDim,xDim,zDim)];
}

//mono index cell representation
bool CA3D::getSubstateValue_BOOL(unsigned int substateLabel,unsigned int index) const{
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==BOOL);
	return ((bool*)substates[substateLabel])[index];
}

double CA3D::getSubstateValue_DOUBLE(unsigned int substateLabel,unsigned int index)const{
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==DOUBLE);
	return ((double*)substates[substateLabel])[index];
}

float CA3D::getSubstateValue_FLOAT(unsigned int substateLabel,unsigned int index)const{
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==FLOAT);
	return ((float*)substates[substateLabel])[index];
}

int CA3D::getSubstateValue_INT(unsigned int substateLabel,unsigned int index)const{
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==INT);
	return ((int*)substates[substateLabel])[index];
}

char CA3D::getSubstateValue_CHAR(unsigned int substateLabel,unsigned int index)const{
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==CHAR);
	return ((char*)substates[substateLabel])[index];
}




/* ------------------END GET SUBSTATE VALUE FAMILY------------------*/


/* ----------------START SET SUBSTATE FAMILY FUNCTION ------------------*/
void CA3D::setSubstateValue3D_BOOL(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k,bool const value) {
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==BOOL);
	((bool*)substates[substateLabel])[getLinearIndex(i,j,k,yDim,xDim,zDim)]=value;
}

void CA3D::setSubstateValue3D_DOUBLE(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k, double const value){
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==DOUBLE);
	((double*)substates[substateLabel])[getLinearIndex(i,j,k,yDim,xDim,zDim)]=value;
}

void CA3D::setSubstateValue3D_FLOAT(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k,float const value){
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==FLOAT);
	((float*)substates[substateLabel])[getLinearIndex(i,j,k,yDim,xDim,zDim)]=value;
}

void CA3D::setSubstateValue3D_INT(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k,int const value){
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==INT);
	((int*)substates[substateLabel])[getLinearIndex(i,j,k,yDim,xDim,zDim)]=value;
}

void CA3D::setSubstateValue3D_CHAR(unsigned int substateLabel,unsigned int i, unsigned int j,unsigned int k,char const value){
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==CHAR);
	((char*)substates[substateLabel])[getLinearIndex(i,j,k,yDim,xDim,zDim)]=value;
}


void CA3D::setSubstateValue_BOOL(unsigned int substateLabel,unsigned int index,bool const value) {
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==BOOL);
	((bool*)substates[substateLabel])[index]=value;
}

void CA3D::setSubstateValue_DOUBLE(unsigned int substateLabel,unsigned int index, double const value){
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==DOUBLE);
	((double*)substates[substateLabel])[index]=value;
}

void CA3D::setSubstateValue_FLOAT(unsigned int substateLabel,unsigned int index,float const value){
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==FLOAT);
	((float*)substates[substateLabel])[index]=value;
}

void CA3D::setSubstateValue_INT(unsigned int substateLabel,unsigned int index,int const value){
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==INT);
	((int*)substates[substateLabel])[index]=value;
}

void CA3D::setSubstateValue_CHAR(unsigned int substateLabel,unsigned int index,char const value){
	assert(substateLabel<=substate_count);
	assert(substateTypes[substateLabel]==CHAR);
	((char*)substates[substateLabel])[index]=value;
}


/* ------------------END SET SUBSTATE VALUE FAMILY------------------*/



void CA3D::registerStopCondictionCallback(bool(*stopCondition_callback)()){
	assert(stopCondition_callback!=NULL);
	stopCondition=stopCondition_callback;
}



/*It checks whether or not all the callbacks, substates,
 * matrices parameter are in coherent state.
 * If it works correctly computation may take place
 * Return:
 * 		TRUE if everything is OK
 * 		FALSE stop the automata. Finalize memories-> shutdown
 */
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!IMPLEMENTALA BENEEEEEEEEEEEEEEEEEEEEEEEEEEE
bool CA3D::checkAutomataStatusBeforeComputation(){

	/*at least one substate and one callback have to be provided*/

	/*substate number parameter has to match the substate actually added*/

	/*function callbacks number parameter has to match the callbacks actually registered*/

	/*A stop function has to be provided as callback*/
	return true;
}




void CA3D::globalTransitionFunction_MAINLOOP(){
	clock_t start = clock();


	/*------------------------------------------------------------------------------*/

	unsigned int k=0;
	while(!stop){
		//for each elementary process
		for(k=0;k<elementaryProcesses_size;k++){
			//printf("elementaryProcess -> %i\n",k);
			//loops over all cells of the cellular automata

			(elementaryProcesses[k])<<<dimGrid,blockDim>>>(d_CA);
			cudaThreadSynchronize();


		}
		//printf("DIMGRID(%i,%i,%i), BlockDim(%i,%i,%i)\n",dimGrid.x,dimGrid.y,dimGrid.z,blockDim.x,blockDim.y,blockDim.z);

		steps=steps+1;
		printf("Step = %i\n",steps);
		stop=stopCondition();

		//callback each
		if(steps%stepsBetweenCallback==0){
			//callback occurs
			callback(steps);

		}

	}

	/*-----------------------------------------------------------------------------------*/
	clock_t end = clock();
	elapsedTime = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Step performed = %i\nElapsed Time=%.4f\n",steps,elapsedTime);

}

void CA3D::globalTransitionFunction(){
	if(!checkAutomataStatusBeforeComputation()){
		//error are printed out by the function checkAutomataStatusBeforeComputation() directly
		cleanup();
		exit(-1);
	}
	globalTransitionFunction_MAINLOOP();
}

void CA3D::registerElementaryProcess( void(*callback)(CA_GPU3D* d_CA ) ){
	assert(callback!=NULL && elementaryProcesses_count < elementaryProcesses_size );
	elementaryProcesses[elementaryProcesses_count]=callback;
	elementaryProcesses_count++;
}


void CA3D::setInitialParameters(unsigned int substates_size,unsigned int transitionFunction_size){
	/**
	 * substates_size = The number of substates of the automaton
	 * transitionFunction_size = The number of transition functions
	 * */
	this->substates_size=substates_size;
	this->elementaryProcesses_size=transitionFunction_size;
}

/**
 * It does preliminary allocation of substates arrays
 * and function callbacks for elementary processes
 * */
void CA3D::initialize(){
	assert(elementaryProcesses==0 && substates==0 && substateTypes==0);
	elementaryProcesses=(void(**)(CA_GPU3D*))malloc(sizeof(void(*)(CA_GPU3D*))*elementaryProcesses_size);
	substates= (void**)malloc(sizeof(void*)*substates_size);
	substateTypes =(TYPE*)malloc(sizeof(TYPE)*substates_size);
}



void CA3D::cleanup(){
	printDebug("CLEANUP - START");
	unsigned int i=0;
	//free all the allocated substates
	for(;i<substate_count;i++){
		free(substates[i]);
		printDebug("FREED");
	}

	//free(elementaryProcesses);//it is allocated on GPU
	free(substates);
	printDebug("CLEANUP - END");
}

void CA3D::addSubstate(SUBSTATE_LABEL label,TYPE t){

	void * substate=NULL;
	substate=allocateSubstate(t,substate);
	registerSubstate(substate,label,t);

}


void CA3D::registerSubstate(void * buffer,SUBSTATE_LABEL label,TYPE t){
	assert( (substate_count < (substates_size)) && (buffer != NULL) && (label < (substates_size)) );
	substates[label]=buffer;
	substateTypes[label]=t;
	substate_count++;

}

void* CA3D::allocateSubstate(TYPE t,void* buffer){
	switch(t){

	case FLOAT:
		buffer = (float*)malloc(numCells*sizeof(float));
		break;
	case DOUBLE:
		buffer = (double*)malloc(numCells*sizeof(double));
		break;
	case CHAR:
		buffer = (char*)malloc(numCells*sizeof(char));
		break;
	case INT:
		buffer = (int*)malloc(numCells*sizeof(int));
		break;
	case BOOL:
		buffer = (bool*)malloc(numCells*sizeof(bool));
		break;

	}
	//map the correnspondent buffer just created to its type


	return buffer;
}

void CA3D::updateDimGrid(){
	dimGrid.x= (xDim/blockDim.x)+(xDim%blockDim.x == 0 ? 0 : 1);
	dimGrid.y= (yDim/blockDim.y)+(yDim%blockDim.y == 0 ? 0 : 1);
	dimGrid.z= (zDim/blockDim.z)+(zDim%blockDim.z == 0 ? 0 : 1);
}



/*GET i-th NEIGHBOR INDEX functions MOORE NEIGHBORHOOD

	         5 | 1 | 8
	        ---|---|---
	         2 | 0 | 3
	        ---|---|---
	         6 | 4 | 7
 */
unsigned int CA3D::getNeighborIndex3D_MOORE(unsigned int i, unsigned int j,unsigned int k,unsigned int neighbor){
	assert(neighbor<9);
	switch(neighbor){
	case 0:
		return getLinearIndex(i,j,k,yDim,xDim,zDim);
	case 1:
		return getLinearIndex(i,j,k,yDim,xDim,zDim);//one row up
	case 2:
		return getLinearIndex(i,j-1,k,yDim,xDim,zDim);//same row one coloumn left
	case 3:
		return getLinearIndex(i,j+1,k,yDim,xDim,zDim);//same row one coloumn right
	case 4:
		return getLinearIndex(i+1,j,k,yDim,xDim,zDim);//same column one row down
	case 5:
		return getLinearIndex(i-1,j-1,k,yDim,xDim,zDim);//one row up one col left
	case 6:
		return getLinearIndex(i+1,j-1,k,yDim,xDim,zDim);//one row down one col left
	case 7:
		return getLinearIndex(i+1,j+1,k,yDim,xDim,zDim);//row down col right
	case 8:
		return getLinearIndex(i-1,j+1,k,yDim,xDim,zDim);//row up col right
	}

	return NULL;//it should never be executed


}
unsigned int CA3D::getNeighborIndex3D_MOORE(unsigned int index,unsigned int neighbor){
	assert(neighbor<9);
	switch(neighbor){
	case 0:
		return index;
	case 1:
		return index-xDim;//one row up
	case 2:
		return index-1;//same row one coloumn left
	case 3:
		return index+1;//same row one coloumn right
	case 4:
		return index+xDim;//same column one row down
	case 5:
		return index-xDim-1;//one row up one col left
	case 6:
		return index+xDim-1;//one row down one col left
	case 7:
		return index+xDim+1;//row down col right
	case 8:
		return index-xDim+1;//row up col right
	}

	return NULL;//it should never be executed
}





//GETTER AND SETTER
unsigned int CA3D::get_xDim() const {
	return xDim;
}

unsigned int CA3D::getElementaryProcessesSize() const {
	return elementaryProcesses_size;
}

unsigned int CA3D::get_yDim() const {
	return yDim;
}

unsigned int CA3D::getSubstatesSize() const {
	return substates_size;
}



unsigned int CA3D::getBlockdimX() const {
	return blockDim.x;
}

void CA3D::setBlockdimX(unsigned int dimX) {
	if(isPowerOfTwo(dimX)){
		this->blockDim.x = dimX;
	}else{
		printf("WARNING -> setBlockDimX has to be power of 2 -> dimX=%i",DEFAULT_BLOCKDIM_X);
		blockDim.x=DEFAULT_BLOCKDIM_X;
	}
	updateDimGrid();


}

unsigned int CA3D::getBlockDimY() const {
	return this->blockDim.y;
}

void CA3D::setBlockDimY(unsigned int dimY) {
	if(isPowerOfTwo(dimY)){
		this->blockDim.y = dimY;
	}else{
		printf("WARNING -> setBlockDimY has to be power of 2 -> dimY=%i",DEFAULT_BLOCKDIM_Y);
		blockDim.y=DEFAULT_BLOCKDIM_Y;
	}
	updateDimGrid();
}

void CA3D::setBlockdimZ(unsigned int dimZ){
	if(isPowerOfTwo(dimZ)){
		this->blockDim.z = dimZ;
	}else{
		printf("WARNING -> setBlockDimZ has to be power of 2 -> dimZ=%i",DEFAULT_BLOCKDIM_Z);
		blockDim.z=DEFAULT_BLOCKDIM_Z;
	}
	updateDimGrid();
}

unsigned int CA3D::getBlockdimZ() const{
	return this->blockDim.z;
}

unsigned int CA3D::getStepsBetweenCopy() const {
	return stepsBetweenCallback;
}





void CA3D::setStepsBetweenCopy(unsigned int stepsBetweenCopy) {
	this->stepsBetweenCallback = stepsBetweenCopy;
}

unsigned int CA3D::isPowerOfTwo (unsigned int x)
{
	unsigned int powerOfTwo = 1;

	while (powerOfTwo < x && powerOfTwo < 2147483648)
		powerOfTwo *= 2;
	return (x == powerOfTwo);
}


//END OFGETTER AND SETTER

void CA3D::copyBuffersFromGPU(){
	printDebug("START offload copy");
	for(int i=0;i<substates_size;i++){
		copyBufferFromGPU(substates[i],d_subPointer[i],substateTypes[i]);
	}
	printDebug("END offload copy");
}

/**
 * Set default parameters and do normal initialization
 * */
void CA3D::preliminaryCAConstructor() {
	assert(yDim <= 0);
	this->steps=0;
	this->elapsedTime=0.0f;

	substates=NULL;
	substates_size=0;
	substate_count=0;
	substateTypes=0;
	stopCondition=0;
	stop=false;//global transition func main loop ACTIVE

	elementaryProcesses=NULL;
	elementaryProcesses_size=0;
	elementaryProcesses_count=0;

}

//2D constructor
CA3D::CA3D(int YDim,int XDim,int ZDim,bool toroidal){
	this->preliminaryCAConstructor();
	this->yDim=YDim;
	this->xDim=XDim;
	this->zDim=ZDim;
	this->numCells=yDim*xDim*zDim;
	this->isToroidal=toroidal;
	if(isToroidal){
		getLinearIndex=getLinearIndexToroidal3D;
	}else{
		getLinearIndex=getLinearIndexNormal3D;
	}
	blockDim.x=DEFAULT_BLOCKDIM_X;
	blockDim.y=DEFAULT_BLOCKDIM_Y;
	blockDim.z=DEFAULT_BLOCKDIM_Z;
	updateDimGrid();

}


void CA3D::setCallback(void(*call)(unsigned int)){
	this->callback=call;
}


