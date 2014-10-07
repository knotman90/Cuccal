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
#include <cstring>
#include <iostream>
#include "CAGLVisualizer.h"
//CA dichiarata in CA.h
extern CA CA;


/* Handler for window re-size event. Called back when the window first appears and
   whenever the window is re-sized with its new width and height */
void reshape(GLsizei width, GLsizei height) {  // GLsizei for non-negative integer
	// Compute aspect ratio of the new window
	if (height == 0) height = 1;                // To prevent divide by 0
	GLfloat aspect = (GLfloat)width / (GLfloat)height;

	// Set the viewport to cover the new window
	glViewport(0, 0, width, height);

	// Set the aspect ratio of the clipping area to match the viewport
	glMatrixMode(GL_PROJECTION);  // To operate on the Projection matrix
	glLoadIdentity();             // Reset the projection matrix
	if (width >= height) {
		// aspect >= 1, set the height from -1 to 1, with larger width
		gluOrtho2D(-1.0 * aspect, 1.0 * aspect, -1.0, 1.0);
	} else {
		// aspect < 1, set the width to -1 to 1, with larger height
		gluOrtho2D(-1.0, 1.0, -1.0 / aspect, 1.0 / aspect);
	}
}

//Graphical Callback
void initializeVisualizer(int argc, char** argv){
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(500,500);
	glutInitWindowPosition(100,100);
	glutCreateWindow("OpenGL - First window demo");
	glutReshapeFunc(reshape);
}



void renderFunction(){
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0, 1.0, 1.0);
	glMatrixMode(GL_PROJECTION);      // Select the Projection matrix for operation
	glLoadIdentity();                 // Reset Projection matrix
	gluOrtho2D(0, CA.cols, -300, 0);
	for(float i=0;i<CA.rows;i++){
		for(float j=0;j<CA.cols;j++){
			if(CA.getSubstateValue_BOOL(Q,i,j)){
				glColor3f(1.0, 1.0, 0.0);
			}else{
				glColor3f(1.0, 0.0, 1.0);
			}
			glBegin(GL_POLYGON);
			glVertex2f(j+1, -i+1);
			glVertex2f(j+1, -i);
			glVertex2f(j, -i);
			glVertex2f(j, -i+1);
			glEnd();

		}

	}

	glutSwapBuffers();
	//glutPostRedisplay();
	glFlush();

}






void callback(unsigned int currentsteps){
	char path[20];
	sprintf(path, "Q_%d.sst", currentsteps);
	CA.copyBufferFromGPU(CA.substates[Q],CA.d_subPointer[Q],CA.substateTypes[Q]);
	//glutPostRedisplay();
	//respawn graphicThread and draw substate


	//CA.saveSubstate(Q,path);
}



//mod 2 automaton
//__global__ void gpuEvolve(CA_GPU* d_CA){
//	unsigned int col=(threadIdx.x+blockIdx.x*blockDim.x);
//	unsigned int row=(threadIdx.y+blockIdx.y*blockDim.y);
//	unsigned int totRows=d_CA->scalars->rows;
//	unsigned int totCols=d_CA->scalars->cols;
//	if(row<totRows && col<totCols){
//		short unsigned int count=0;
//		unsigned int linNeighIdx=0;
//		bool alive=d_CA->getSubstateValue_BOOL(Q,row,col);
//		for (int neigh = 1; neigh < 9; neigh++) {
//			linNeighIdx=d_CA->getNeighborIndex_MOORE_Toroidal(row,col,neigh,totRows,totCols);
//			if(d_CA->getSubstateValue_BOOL(Q,linNeighIdx)==true){
//				count++;
//			}
//		}
//		alive=alive%2==0 ? true : false;
//		d_CA->setSubstateValue_BOOL(Q_NEW,row,col,alive);
//	}
//
//}

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

	if(CA.getSteps()>10000){
		return true;
	}
	return false;
}


