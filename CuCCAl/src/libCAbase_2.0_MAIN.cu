//============================================================================
// Name        : 0.cpp
// Author      : Davide Spataro
// Version     :
// Copyright   : 
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "CA2D/CA2D.cuh"
#include "CA3D/CA3D.cuh"
#include "CAGLVisualizer.h"
#include <pthread.h>


//proto functions defined by user in config.cpp for 2D automaton
extern void callback(unsigned int);
extern void gpuEvolve(CA_GPU2D* d_CA);
extern void copyBoard(CA_GPU2D* d_CA);
extern bool stopCondition();

extern void initializeVisualizer(int argc, char** argv);
extern void renderFunction();

extern void displayGOL3D();
extern void initGOL3D();


//proto functions defined by user in config.cpp for 2D automaton
extern void callback3D(unsigned int);
extern void gpuEvolve3D(CA_GPU3D* d_CA);
extern void copyBoard3D(CA_GPU3D* d_CA);
extern bool stopCondition3D();





CA2D CA(401,401,true);
int CA2DMain(int argc, char **argv){
	/*--------START CONFIGURATION AND INITIALIZATION PHASE--------*/
	CThread* visualizer= new CAGLVisualizer(argc, argv);
	((CAGLVisualizer*)visualizer)->setInitializeCallback(initializeVisualizer);
	((CAGLVisualizer*)visualizer)->setRenderCallBack(renderFunction);
	visualizer->Start();

	initializeVisualizer(argc,argv);

	CA.setInitialParameters(2,2);
	CA.initialize();

	CA.addSubstate(Q,BOOL);
	CA.addSubstate(Q_NEW,BOOL);


	CA.registerElementaryProcess(gpuEvolve);
	CA.registerElementaryProcess(copyBoard);
	CA.registerStopCondictionCallback(stopCondition);

	CA.setBlockDimY(16);
	CA.setBlockdimX(16);

	CA.setStepsBetweenCopy(1);
	CA.setCallback(callback);


	if(CA.loadSubstate(Q,"./data/GOL/GOL_400x400.sst")==ERROR_OPENING_FILE){
		printDebug("ERROR opening file");
		return -1;
	}

	if(CA.saveSubstate(Q,"./Q_original.sst")==ERROR_OPENING_FILE){
		printDebug("ERROR SAVING");
	}

	CA.initializeGPUAutomata();


	/*--------END CONFIGURATION AND INITIALIZATION PHASE--------*/
	CA.globalTransitionFunction();

	CA.copyBuffersFromGPU();
	CA.cleanUpGPUAutomata();
	//CA.printSubstate_STDOUT(Q);
	//CA.printSubstate_STDOUT(Q_NEW);

	if(CA.saveSubstate(Q,"./Q.sst")==ERROR_OPENING_FILE){
		printDebug("ERROR SAVING");
	}
	if(CA.saveSubstate(Q_NEW,"./Q_NEW.sst")==ERROR_OPENING_FILE){
		printDebug("ERROR SAVING");
	}



	visualizer->Join();
	CA.cleanup();
	printf("\nElapsed Time = %.5f \nEND",CA.elapsedTime);

}



CA3D CA_3D(50,50,20,false);
CThread* visualizer;
bool* start;
int CA3DMain(int argc, char **argv){

	CA_3D.setInitialParameters(2,2);
	CA_3D.initialize();

	CA_3D.addSubstate(Q,BOOL);
	CA_3D.addSubstate(Q_NEW,BOOL);
	srand(time(0));
	for(int k=0;k<CA_3D.zDim;k++){
		for(int i=0;i<CA_3D.yDim;i++){
			for(int j=0;j<CA_3D.xDim;j++){
				//cout<<"("<<i<<" "<<j<<" "<<k<<": "<<hd_getLinearIndexNormal3D(i,j,k,CA_3D.yDim,CA_3D.xDim,CA_3D.zDim)<<") ";
				if(rand()%100 < 30){
					CA_3D.setSubstateValue_BOOL3D(Q,hd_getLinearIndexNormal3D(i,j,k,CA_3D.yDim,CA_3D.xDim,CA_3D.zDim),true);
				}else{
					CA_3D.setSubstateValue_BOOL3D(Q,hd_getLinearIndexNormal3D(i,j,k,CA_3D.yDim,CA_3D.xDim,CA_3D.zDim),false);
				}
			}
			cout<<endl;
		}
	}

	if(CA_3D.saveSubstate(Q,"./Q3D_original.sst")==ERROR_OPENING_FILE){
		printDebug("ERROR SAVING");
	}

	CA_3D.registerElementaryProcess(gpuEvolve3D);
	CA_3D.registerElementaryProcess(copyBoard3D);
	CA_3D.registerStopCondictionCallback(stopCondition3D);

	CA_3D.setBlockDimY(8);
	CA_3D.setBlockdimX(8);
	CA_3D.setBlockdimZ(8);

	CA_3D.setStepsBetweenCopy(10);
	CA_3D.setCallback(callback);

	CA_3D.initializeGPUAutomata();


	/*--------END CONFIGURATION AND INITIALIZATION PHASE--------*/
	CA_3D.globalTransitionFunction();

	CA_3D.copyBuffersFromGPU();
	CA_3D.cleanUpGPUAutomata();
	CA_3D.printSubstate_STDOUT(Q);
	//CA_3D.printSubstate_STDOUT(Q_NEW);

	if(CA_3D.saveSubstate(Q,"./Q.sst")==ERROR_OPENING_FILE){
		printDebug("ERROR SAVING");
	}
	if(CA_3D.saveSubstate(Q_NEW,"./Q_NEW.sst")==ERROR_OPENING_FILE){
		printDebug("ERROR SAVING");
	}




	CA_3D.cleanup();
	printf("\nElapsed Time = %.5f \nEND",CA.elapsedTime);

}


int main(int argc, char **argv) {
	return CA3DMain(argc,argv);
	//CA2DMain(argc,argv);
	//CADebug();





	printf("EXITING\n");

	return 0;
}





