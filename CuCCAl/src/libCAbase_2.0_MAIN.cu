//============================================================================
// Name        : 0.cpp
// Author      : Davide Spataro
// Version     :
// Copyright   : 
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "CA2D.cuh"
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


CA3D CA_3D(100,100,100,false);
int CA3DMain(int argc, char **argv){
	/*--------START CONFIGURATION AND INITIALIZATION PHASE--------*/
//	CThread* visualizer= new CAGLVisualizer(argc, argv);
//	((CAGLVisualizer*)visualizer)->setInitializeCallback(initializeVisualizer);
//	((CAGLVisualizer*)visualizer)->setRenderCallBack(renderFunction);
//	visualizer->Start();


	CA_3D.setInitialParameters(2,2);
	CA_3D.initialize();

	CA_3D.addSubstate(Q,BOOL);
	CA_3D.addSubstate(Q_NEW,BOOL);


	CA_3D.registerElementaryProcess(gpuEvolve3D);
	CA_3D.registerElementaryProcess(copyBoard3D);
	CA_3D.registerStopCondictionCallback(stopCondition3D);

	CA_3D.setBlockDimY(16);
	CA_3D.setBlockdimX(16);
	CA_3D.setBlockdimZ(16);

	CA_3D.setStepsBetweenCopy(1);
	CA_3D.setCallback(callback);



	CA_3D.initializeGPUAutomata();
//
//
//	/*--------END CONFIGURATION AND INITIALIZATION PHASE--------*/
//	CA.globalTransitionFunction();
//
//	CA.copyBuffersFromGPU();
	CA.cleanUpGPUAutomata();
//	//CA.printSubstate_STDOUT(Q);
//	//CA.printSubstate_STDOUT(Q_NEW);
//
//	if(CA.saveSubstate(Q,"./Q.sst")==ERROR_OPENING_FILE){
//		printDebug("ERROR SAVING");
//	}
//	if(CA.saveSubstate(Q_NEW,"./Q_NEW.sst")==ERROR_OPENING_FILE){
//		printDebug("ERROR SAVING");
//	}
//
//
//
//	visualizer->Join();
	CA.cleanup();
	printf("\nElapsed Time = %.5f \nEND",CA.elapsedTime);

}


int main(int argc, char **argv) {
	return CA3DMain(argc,argv);

	 //CADebug();





	 printf("EXITING\n");

	return 0;
}





