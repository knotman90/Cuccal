//============================================================================
// Name        : 0.cpp
// Author      : Davide Spataro
// Version     :
// Copyright   : 
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "CA.cuh"


//proto functions defined by user in config.cpp
extern void gpuEvolve(CA_GPU* d_CA);
extern void copyBoard(CA_GPU* d_CA);
extern bool stopCondition();



CA CA(401,401,true);



int main() {
	/*--------START CONFIGURATION AND INITIALIZATION PHASE--------*/



	CA.setInitialParameters(2,2);
	CA.initialize();

	CA.addSubstate(Q,BOOL);
	CA.addSubstate(Q_NEW,BOOL);



	CA.registerElementaryProcess(gpuEvolve);
	CA.registerElementaryProcess(copyBoard);
	CA.registerStopCondictionCallback(stopCondition);

	CA.setBlockDimY(16);
	CA.setBlockdimX(16);




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




	CA.cleanup();
	printf("\nElapsed Time = %.5f \nEND",CA.elapsedTime);




	return 0;
}





