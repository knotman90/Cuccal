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
#include "CA2D/CA2D.cuh"
#include "CA3D/CA3D.cuh"
#include <cstring>
#include <iostream>
#include "CAGLVisualizer.h"
//CA dichiarata in CA.h
extern CA2D CA;
extern CA3D CA_3D;

void initGOL3D(void);

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

extern bool*start;
//Graphical Callback
void initializeVisualizer(int argc, char** argv){
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(640, 480);
	glutInitWindowPosition(100,100);
	glutCreateWindow("OpenGL - First window demo");
	glutReshapeFunc(reshape);

	initGOL3D();
	*start=true;

	printf("CHIAMATA GLUT INITTTTTT\n");


}



void renderFunction(){
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0, 1.0, 1.0);
	glMatrixMode(GL_PROJECTION);      // Select the Projection matrix for operation
	glLoadIdentity();                 // Reset Projection matrix
	gluOrtho2D(0, CA.xDim, -300, 0);
	for(float i=0;i<CA.yDim;i++){
		for(float j=0;j<CA.xDim;j++){
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





extern CThread* visualizer;
void callback(unsigned int currentsteps){
	//CA_3D.copyBufferFromGPU(CA_3D.substates[Q],CA_3D.d_subPointer[Q],CA_3D.substateTypes[Q]);
	//glutPostRedisplay();
}



//mod 2 automaton
//__global__ void gpuEvolve(CA_GPU* d_CA){
//	unsigned int col=(threadIdx.x+blockIdx.x*blockDim.x);
//	unsigned int row=(threadIdx.y+blockIdx.y*blockDim.y);
//	unsigned int totRows=d_CA->scalars->rows;
//	unsigned int totCols=d_CA->scalars->xDim;
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

__global__ void gpuEvolve(CA_GPU2D* d_CA){
	unsigned int col=(threadIdx.x+blockIdx.x*blockDim.x);
	unsigned int row=(threadIdx.y+blockIdx.y*blockDim.y);
	unsigned int totRows=d_CA->scalars->yDim;
	unsigned int totCols=d_CA->scalars->xDim;
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


void __global__ copyBoard(CA_GPU2D* d_CA){
	int col=(threadIdx.x+blockIdx.x*blockDim.x);
	int row=(threadIdx.y+blockIdx.y*blockDim.y);
	if(row<d_CA->scalars->yDim && col<d_CA->scalars->xDim){
		d_CA->setSubstateValue_BOOL(Q,row,col,d_CA->getSubstateValue_BOOL(Q_NEW,row,col));
	}

}




//true means --> STOP THE AUTOMATA
bool stopCondition(){

	if(CA.getSteps()>10){
		return true;
	}
	return false;
}





//3D functions CA definition
//proto functions defined by user in config.cpp for 2D automaton

void callback3D(unsigned int currentsteps){

	printf("callback 3D %d", currentsteps);

}


////mod 2 automaton
//__global__ void gpuEvolve3D(CA_GPU3D* d_CA){
//	unsigned int x=(threadIdx.x+blockIdx.x*blockDim.x);
//	unsigned int y=(threadIdx.y+blockIdx.y*blockDim.y);
//	unsigned int z=(threadIdx.z+blockIdx.z*blockDim.z);
//	unsigned int dimY=d_CA->scalars->yDim;
//	unsigned int dimX=d_CA->scalars->xDim;
//	unsigned int dimZ=d_CA->scalars->zDim;
//	if(y<dimY && x<dimX && z < dimZ){
//		short unsigned int count=0;
//		unsigned int linNeighIdx=0;
//		bool alive=d_CA->getSubstateValue_BOOL(Q,y,x,z);
//		for (int neigh = 1; neigh < MOORE_NEIGHBORHOOD ; neigh++) {
//			linNeighIdx=d_CA->getNeighborIndex_MOORE_Toroidal(y,x,z,neigh,dimY,dimX,dimZ);
//			if(d_CA->getSubstateValue_BOOL(Q,linNeighIdx)==true){
//				count++;
//			}
//		}
//		alive=count%2==0 ? true : false;
//		d_CA->setSubstateValue_BOOL(Q_NEW,y,x,z,alive);
//	}
//
//}

//gol3d  automaton
__global__ void gpuEvolve3D(CA_GPU3D* d_CA){
	unsigned int x=(threadIdx.x+blockIdx.x*blockDim.x);
	unsigned int y=(threadIdx.y+blockIdx.y*blockDim.y);
	unsigned int z=(threadIdx.z+blockIdx.z*blockDim.z);
	unsigned int dimY=d_CA->scalars->yDim;
	unsigned int dimX=d_CA->scalars->xDim;
	unsigned int dimZ=d_CA->scalars->zDim;
	if(y<dimY && x<dimX && z < dimZ){
		short unsigned int count=0;
		unsigned int linNeighIdx=0;
		bool alive=d_CA->getSubstateValue_BOOL(Q,y,x,z);
		for (int neigh = 1; neigh < MOORE_NEIGHBORHOOD ; neigh++) {
			linNeighIdx=d_CA->getNeighborIndex_MOORE_Toroidal(y,x,z,neigh,dimY,dimX,dimZ);
			if(d_CA->getSubstateValue_BOOL(Q,linNeighIdx)==true){
				count++;
			}
		}
		alive=((!alive && count==4) || (alive && ( count==5 || count==6))) ? true : false;
		d_CA->setSubstateValue_BOOL(Q_NEW,y,x,z,alive);
	}

}

void __global__ copyBoard3D(CA_GPU3D* d_CA){
	int x=(threadIdx.x+blockIdx.x*blockDim.x);
	int y=(threadIdx.y+blockIdx.y*blockDim.y);
	int z=(threadIdx.z+blockIdx.z*blockDim.z);
	if(y<d_CA->scalars->yDim && x<d_CA->scalars->xDim && z < d_CA->scalars->zDim){
		d_CA->setSubstateValue_BOOL(Q,y,x,z,d_CA->getSubstateValue_BOOL(Q_NEW,y,x,z));
	}

}




//true means --> STOP THE AUTOMATA
bool stopCondition3D(){

	if(CA_3D.getSteps()>50){
		return true;
	}
	return false;
}



//GAme of Life 3D!--------------------------------------------------------------------------------

struct ModelView{
	GLfloat x_rot;
	GLfloat y_rot;
	GLfloat z_trans;
};

struct ModelView model_view;






void displayGOL3D()
{
	unsigned int i,j,k;
	bool state;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPushMatrix();

	glTranslatef(0, 0, model_view.z_trans);
	glRotatef(model_view.x_rot, 1, 0, 0);
	glRotatef(model_view.y_rot, 0, 1, 0);

	// Save the lighting state variables
	glPushAttrib(GL_LIGHTING_BIT);
	glDisable(GL_LIGHTING);
	glPushMatrix();
	glColor3f(0,1,0);
	glScalef(CA_3D.yDim, CA_3D.xDim, CA_3D.zDim);
	glutWireCube(1.0);
	glPopMatrix();
	// Restore lighting state variables
	glPopAttrib();

	glColor3f(1,1,1);
	for (k=0; k<CA_3D.zDim; k++)
		for (i=0; i<CA_3D.yDim; i++)
			for (j=0; j<CA_3D.xDim; j++)
			{
				state = CA_3D.getSubstateValue_BOOL3D(Q,i,j,k);
				if (state)
				{
					glPushMatrix();
					glTranslated(i-CA_3D.yDim/2,j-CA_3D.xDim/2,k-CA_3D.zDim/2);
					glutSolidCube(1.0);
					glPopMatrix();
				}
			}

	glPopMatrix();
	glutSwapBuffers();
}




void reshapeGOL3D(int w, int h)
{
	GLfloat	 lightPos[]	= { 0.0f, 0.0f, 100.0f, 1.0f };
	int MAX = CA_3D.yDim;

	if (MAX < CA_3D.xDim)
		MAX = CA_3D.xDim;
	if (MAX < CA_3D.zDim)
		MAX = CA_3D.zDim;

	glViewport (0, 0, (GLsizei) w, (GLsizei) h);
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity ();
	gluPerspective(45.0, (GLfloat) w/(GLfloat) h, 1.0, 4*MAX);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt (0.0, 0.0, 2*MAX, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

	lightPos[2] = 2*MAX;
	glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
}



void specialKeysGOL3D(int key, int x, int y){

	GLubyte specialKey = glutGetModifiers();
	const GLfloat x_rot = 5.0, y_rot = 5.0, z_trans = 5.0;

	if(key==GLUT_KEY_DOWN){
		model_view.x_rot += x_rot;
	}
	if(key==GLUT_KEY_UP){
		model_view.x_rot -= x_rot;
	}
	if(key==GLUT_KEY_LEFT){
		model_view.y_rot -= y_rot;
	}
	if(key==GLUT_KEY_RIGHT){
		model_view.y_rot += y_rot;
	}
	if(key == GLUT_KEY_PAGE_UP){
		model_view.z_trans += z_trans;
	}
	if(key == GLUT_KEY_PAGE_DOWN){
		model_view.z_trans -= z_trans;
	}

	glutPostRedisplay();
}

void initGOL3D()
{
	GLfloat  ambientLight[] = { 0.2f, 0.2f, 0.2f, 1.0f };
	GLfloat  diffuseLight[] = { 0.75f, 0.75f, 0.75f, 1.0f };

	glEnable(GL_LIGHTING);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambientLight);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,  diffuseLight);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);

	glClearColor (0.0, 0.0, 0.0, 0.0);
	glShadeModel (GL_FLAT);

	glEnable (GL_DEPTH_TEST);

	model_view.x_rot = 0.0;
	model_view.y_rot = 0.0;
	model_view.z_trans = 0.0;

	glutDisplayFunc(displayGOL3D);
	glutReshapeFunc(reshapeGOL3D);
	glutSpecialFunc(specialKeysGOL3D);

	printf("The life 3D cellular automata model\n");
	printf("Left click on the graphic window to start the simulation\n");
	printf("Right click on the graphic window to stop the simulation\n");
}
