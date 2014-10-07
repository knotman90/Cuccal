/*
 * CAGLVisualizer.cpp
 *
 *  Created on: 07/ott/2014
 *      Author: knotman
 */

#include "CAGLVisualizer.h"



CAGLVisualizer::CAGLVisualizer(int argC, char**argV) {
	initialized=false;
	this->argc=argC;
	this->argv=argV;

}

CAGLVisualizer::~CAGLVisualizer() {
	// TODO Auto-generated destructor stub
}



void* CAGLVisualizer::graphicThreadEntryPoint(void*){

			glutMainLoop();
}

//void CAGLVisualizer::initialize(int argC, char **argV) {
//	if(!initialized){
//		this->argc=argC;
//		this->argv=argV;
//		glutInit(&argc,argv);
//		glutInitDisplayMode(GLUT_SINGLE);
//		glutInitWindowSize(500,500);
//		glutInitWindowPosition(100,100);
//		glutCreateWindow("OpenGL - First window demo");
//		glutDisplayFunc(this->renderFunction);
//
//
//	}else{
//		printDebug("GL environment already initialized");
//	}
//
//}

//void CAGLVisualizer::renderFunction() {
//
//	glClearColor(0.0, 0.0, 0.0, 0.0);
//	    glClear(GL_COLOR_BUFFER_BIT);
//	    glColor3f(1.0, 1.0, 1.0);
//	    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
//	    glBegin(GL_POLYGON);
//	        glVertex2f(-0.5, -0.5);
//	        glVertex2f(-0.5, 0.5);
//	        glVertex2f(0.5, 0.5);
//	        glVertex2f(0.5, -0.5);
//	    glEnd();
//	    glFlush();
//
//}
