//#ifndef CONFIG_GOL3D_CU_
//#define CONFIG_GOL3D_CU_
//
//#include <stdio.h>
//#include <stdlib.h>
//
////3D functions CA definition
////proto functions defined by user in config.cpp for 2D automaton
//
//void callback3D(unsigned int currentsteps){
//
//	printf("callback 3D %d", currentsteps);
//
//}
//
//
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
//		for (int neigh = 1; neigh < 9; neigh++) {
//			linNeighIdx=d_CA->getNeighborIndex_MOORE_Toroidal(y,x,z,neigh,dimY,dimX,dimZ);
//			if(d_CA->getSubstateValue_BOOL(Q,linNeighIdx)==true){
//				count++;
//			}
//		}
//		alive=alive%2==0 ? true : false;
//		d_CA->setSubstateValue_BOOL(Q_NEW,y,x,z,alive);
//	}
//
//}
//
//
//
//void __global__ copyBoard3D(CA_GPU3D* d_CA){
//	int x=(threadIdx.x+blockIdx.x*blockDim.x);
//	int y=(threadIdx.y+blockIdx.y*blockDim.y);
//	int z=(threadIdx.z+blockIdx.z*blockDim.z);
//	if(y<d_CA->scalars->yDim && x<d_CA->scalars->xDim && z < d_CA->scalars->zDim){
//		d_CA->setSubstateValue_BOOL(Q,y,x,z,d_CA->getSubstateValue_BOOL(Q_NEW,y,x,z));
//	}
//
//}
//
//
//
//
////true means --> STOP THE AUTOMATA
//bool stopCondition3D(){
//
//	if(CA.getSteps()>10000){
//		return true;
//	}
//	return false;
//}
//
//
//
////GAme of Life 3D!--------------------------------------------------------------------------------
//
//struct ModelView{
//	GLfloat x_rot;
//	GLfloat y_rot;
//	GLfloat z_trans;
//};
//
//struct ModelView model_view;
//
//
//
//
//
//void displayGOL3D(void)
//{
//	unsigned int i,j,k;
//	bool state;
//
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//	glPushMatrix();
//
//	glTranslatef(0, 0, model_view.z_trans);
//	glRotatef(model_view.x_rot, 1, 0, 0);
//	glRotatef(model_view.y_rot, 0, 1, 0);
//
//	// Save the lighting state variables
//	glPushAttrib(GL_LIGHTING_BIT);
//	glDisable(GL_LIGHTING);
//	glPushMatrix();
//	glColor3f(0,1,0);
//	glScalef(CA_3D.yDim, CA_3D.xDim, CA_3D.zDim);
//	glutWireCube(1.0);
//	glPopMatrix();
//	// Restore lighting state variables
//	glPopAttrib();
//
//	glColor3f(1,1,1);
//	for (k=0; k<CA_3D.zDim; k++)
//		for (i=0; i<CA_3D.yDim; i++)
//			for (j=0; j<CA_3D.xDim; j++)
//			{
//				state = CA_3D.getSubstateValue_BOOL3D(Q,i,j,k);
//				if (state)
//				{
//					glPushMatrix();
//					glTranslated(i-CA_3D.yDim/2,j-CA_3D.xDim/2,k-CA_3D.zDim/2);
//					glutSolidCube(1.0);
//					glPopMatrix();
//				}
//			}
//
//	glPopMatrix();
//	glutSwapBuffers();
//}
//
//
//
//
//void reshapeGOL3D(int w, int h)
//{
//	GLfloat	 lightPos[]	= { 0.0f, 0.0f, 100.0f, 1.0f };
//	int MAX = CA_3D.yDim;
//
//	if (MAX < CA_3D.xDim)
//		MAX = CA_3D.xDim;
//	if (MAX < CA_3D.zDim)
//		MAX = CA_3D.zDim;
//
//	glViewport (0, 0, (GLsizei) w, (GLsizei) h);
//	glMatrixMode (GL_PROJECTION);
//	glLoadIdentity ();
//	gluPerspective(45.0, (GLfloat) w/(GLfloat) h, 1.0, 4*MAX);
//	glMatrixMode(GL_MODELVIEW);
//	glLoadIdentity();
//	gluLookAt (0.0, 0.0, 2*MAX, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
//
//	lightPos[2] = 2*MAX;
//	glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
//}
//
//
//
//void specialKeysGOL3D(int key, int x, int y){
//
//	GLubyte specialKey = glutGetModifiers();
//	const GLfloat x_rot = 5.0, y_rot = 5.0, z_trans = 5.0;
//
//	if(key==GLUT_KEY_DOWN){
//		model_view.x_rot += x_rot;
//	}
//	if(key==GLUT_KEY_UP){
//		model_view.x_rot -= x_rot;
//	}
//	if(key==GLUT_KEY_LEFT){
//		model_view.y_rot -= y_rot;
//	}
//	if(key==GLUT_KEY_RIGHT){
//		model_view.y_rot += y_rot;
//	}
//	if(key == GLUT_KEY_PAGE_UP){
//		model_view.z_trans += z_trans;
//	}
//	if(key == GLUT_KEY_PAGE_DOWN){
//		model_view.z_trans -= z_trans;
//	}
//
//	glutPostRedisplay();
//}
//
//void initGOL3D()
//{
//	GLfloat  ambientLight[] = { 0.2f, 0.2f, 0.2f, 1.0f };
//	GLfloat  diffuseLight[] = { 0.75f, 0.75f, 0.75f, 1.0f };
//
//	glEnable(GL_LIGHTING);
//	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambientLight);
//	glLightfv(GL_LIGHT0, GL_DIFFUSE,  diffuseLight);
//	glEnable(GL_LIGHT0);
//	glEnable(GL_COLOR_MATERIAL);
//	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
//
//	glClearColor (0.0, 0.0, 0.0, 0.0);
//	glShadeModel (GL_FLAT);
//
//	glEnable (GL_DEPTH_TEST);
//
//	model_view.x_rot = 0.0;
//	model_view.y_rot = 0.0;
//	model_view.z_trans = 0.0;
//
//	glutDisplayFunc(displayGOL3D);
//	glutReshapeFunc(reshapeGOL3D);
//	glutSpecialFunc(specialKeysGOL3D);
//
//	printf("The life 3D cellular automata model\n");
//	printf("Left click on the graphic window to start the simulation\n");
//	printf("Right click on the graphic window to stop the simulation\n");
//}
//
//
//
//
//#endif
