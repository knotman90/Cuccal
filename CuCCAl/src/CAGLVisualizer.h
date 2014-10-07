/*
 * GameOflifeGLVisualizer.h
 *
 *  Created on: 07/ott/2014
 *      Author: knotman
 */


#ifndef GAMEOFLIFEGLVISUALIZER_H_
#define GAMEOFLIFEGLVISUALIZER_H_

#include <GL/freeglut.h>
#include <GL/gl.h>
#include "debugUtilities.h"
#include <pthread.h>
#include "CThread.h"
#include <stdlib.h>
#include <stdio.h>




class CAGLVisualizer : public CThread {
//private:
	bool initialized;
	void(*initialize)(int, char**);
	void(*render)();

	int argc;
	char** argv;


	bool paused;
	 void renderMainFunction(){
		while(!paused){
			render();
		}
	}


public:
	CAGLVisualizer(int , char**);
	virtual ~CAGLVisualizer();

	virtual void Run(){
		initialize(argc,argv);
		glutDisplayFunc(render);
		glutMainLoop();

	}


	static void renderFunction();
	static void* graphicThreadEntryPoint(void*);


	void setInitializeCallback(void(*initGLCallback)(int, char**))
	{
		this->initialize = initGLCallback;
	}

	void setRenderCallBack(void(*renderGLCallback)(void) )
	{
		this->render=renderGLCallback;
	}
};

#endif /* GAMEOFLIFEGLVISUALIZER_H_ */
