/*
 * IO_Utils.cpp
 *
 *  Created on: 21/mar/2014
 *      Author: Donato D'ambrosio, Davide Spataro
 *
 */


#include "IO_Utils.h"
using std::cout;
using std::endl;
using std::ios;
//2D automata io function implementation------------------------------------------------------------
template <typename type>
void CA_print_STDOUT2D(const type* const Q,unsigned int yDim,unsigned int xDim){
	cout<<"START-----------------"<<endl;
	for(int i=0;i<yDim;i++){
		for(int j=0;j<xDim;j++){
			cout<<Q[hd_getLinearIndexNormal2D(i,j,yDim,xDim)]<<" ";
		}
		cout<<endl;
	}
	cout<<"END-----------------"<<endl;
}


/* -----------save/load -> to/from FILES  */
template <class type>
void CA_load_substate_FILE2D(fstream& f, type* const Q,unsigned int yDim,unsigned int xDim)
{

	for (int i=0; i<yDim; i++){
		for (int j=0; j<xDim; j++){
			f >> Q[hd_getLinearIndexNormal2D(i,j,yDim,xDim)];
		}
	}

}



template <class type>
int CA_load_substate_FILE2D(const char* const path, type* const Q,unsigned int yDim,unsigned int xDim)
{
	fstream f ( path, ios::in );
	if ( !f.is_open() )
		return ERROR_OPENING_FILE;

	CA_load_substate_FILE2D(f, Q,yDim,xDim);

	f.close();

	return SUCCESS_OPENING_FILE;
}

template <class type>
void CA_save_substate_FILE2D(fstream& f, const type* const Q,unsigned int yDim,unsigned int xDim)
{

	for (int i=0; i<yDim; i++) {
		for (int j=0; j<xDim; j++) {
			f << Q[hd_getLinearIndexNormal2D(i,j,yDim,xDim)] << " ";
		}
		f << "\n";
	}
}


template <class type>
int CA_save_substate_FILE2D(const char* const path, const type* const Q,unsigned int yDim,unsigned int xDim)
{
	fstream f ( path, ios::out );
	if ( !f.is_open() )
		return ERROR_OPENING_FILE;

	CA_save_substate_FILE2D(f, Q,yDim,xDim);

	f.close();

	return SUCCESS_OPENING_FILE;
}

/* END -----------save/load -> to/from FILES*/




//3D automaton IO function implementation---------------------------------------------------------------------------------------

template <typename type>
void CA_print_STDOUT3D(const type* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim){
	cout<<"START-----------------"<<endl;
	for(int k=0;k<zDim;k++){
		for(int i=0;i<yDim;i++){
			for(int j=0;j<xDim;j++){
				cout<<Q[hd_getLinearIndexNormal3D(i,j,k,yDim,xDim,zDim)]<<" ";
			}
			cout<<endl;
		}
	}
	cout<<"END-----------------"<<endl;
}


/* -----------save/load -> to/from FILES  */
template <class type>
void CA_load_substate_FILE3D(fstream& f, type* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim)
{
	for(int k=0;k<zDim;k++){
		for (int i=0; i<yDim; i++){
			for (int j=0; j<xDim; j++){
				f >> Q[hd_getLinearIndexNormal3D(i,j,k,yDim,xDim,zDim)];
			}
		}
	}

}



template <class type>
int CA_load_substate_FILE3D(const char* const path, type* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim)
{
	fstream f ( path, ios::in );
	if ( !f.is_open() )
		return ERROR_OPENING_FILE;

	CA_load_substate_FILE3D(f, Q,yDim,xDim,zDim);

	f.close();

	return SUCCESS_OPENING_FILE;
}

template <class type>
void CA_save_substate_FILE3D(fstream& f, const type* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim)
{

	for(int k=0;k<zDim;k++){
		for (int i=0; i<yDim; i++) {
			for (int j=0; j<xDim; j++) {
				f << Q[hd_getLinearIndexNormal3D(i,j,k,yDim,xDim,zDim)] << " ";
			}
			f << "\n";
		}
	}
}


template <class type>
int CA_save_substate_FILE3D(const char* const path, const type* const Q,unsigned int yDim,unsigned int xDim,unsigned int zDim)
{
	fstream f ( path, ios::out );
	if ( !f.is_open() )
		return ERROR_OPENING_FILE;

	CA_save_substate_FILE3D(f, Q,yDim,xDim,zDim);

	f.close();

	return SUCCESS_OPENING_FILE;
}


//-----------------------------------------------------------------------
//DEFINITION OF THE TEMPLATE FUNCTIONS 2D/3D IO
//-----------------------------------------------------------------------


template void CA_load_substate_FILE2D(fstream& f, bool* const Q,unsigned int yDim,unsigned int xDim);
template void CA_load_substate_FILE2D(fstream& f, double* const Q,unsigned int yDim,unsigned int xDim);
template void CA_load_substate_FILE2D(fstream& f, float* const Q,unsigned int yDim,unsigned int xDim);
template void CA_load_substate_FILE2D(fstream& f, int* const Q,unsigned int yDim,unsigned int xDim);
template void CA_load_substate_FILE2D(fstream& f, char* const Q,unsigned int yDim,unsigned int xDim);

template void CA_save_substate_FILE2D(fstream& f, const bool* const Q,unsigned int yDim,unsigned int xDim);
template void CA_save_substate_FILE2D(fstream& f, const double* const Q,unsigned int yDim,unsigned int xDim);
template void CA_save_substate_FILE2D(fstream& f, const float* const Q,unsigned int yDim,unsigned int xDim);
template void CA_save_substate_FILE2D(fstream& f, const int* const Q,unsigned int yDim,unsigned int xDim);
template void CA_save_substate_FILE2D(fstream& f, const char* const Q,unsigned int yDim,unsigned int xDim);


template int CA_load_substate_FILE2D(const char* const path, double* const Q,unsigned int yDim,unsigned int xDim);
template int CA_load_substate_FILE2D(const char* const path, bool* const Q,unsigned int yDim,unsigned int xDim);
template int CA_load_substate_FILE2D(const char* const path, float* const Q,unsigned int yDim,unsigned int xDim);
template int CA_load_substate_FILE2D(const char* const path, int* const Q,unsigned int yDim,unsigned int xDim);
template int CA_load_substate_FILE2D(const char* const path, char* const Q,unsigned int yDim,unsigned int xDim);


template int CA_save_substate_FILE2D(const char* const path, const bool* const Q,unsigned int yDim,unsigned int xDim);
template int CA_save_substate_FILE2D(const char* const path, const double* const Q,unsigned int yDim,unsigned int xDim);
template int CA_save_substate_FILE2D(const char* const path, const float* const Q,unsigned int yDim,unsigned int xDim);
template int CA_save_substate_FILE2D(const char* const path, const int* const Q,unsigned int yDim,unsigned int xDim);
template int CA_save_substate_FILE2D(const char* const path, const char* const Q,unsigned int yDim,unsigned int xDim);



template void CA_print_STDOUT2D(const double* const Q,unsigned int yDim,unsigned int xDim);
template void CA_print_STDOUT2D(const bool* const Q,unsigned int yDim,unsigned int xDim);
template void CA_print_STDOUT2D(const float* const Q,unsigned int yDim,unsigned int xDim);
template void CA_print_STDOUT2D(const int* const Q,unsigned int yDim,unsigned int xDim);
template void CA_print_STDOUT2D(const char* const Q,unsigned int yDim,unsigned int xDim);


//3D


template void CA_load_substate_FILE3D(fstream& f, bool* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template void CA_load_substate_FILE3D(fstream& f, double* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template void CA_load_substate_FILE3D(fstream& f, float* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template void CA_load_substate_FILE3D(fstream& f, int* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template void CA_load_substate_FILE3D(fstream& f, char* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);

template void CA_save_substate_FILE3D(fstream& f, const bool* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template void CA_save_substate_FILE3D(fstream& f, const double* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template void CA_save_substate_FILE3D(fstream& f, const float* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template void CA_save_substate_FILE3D(fstream& f, const int* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template void CA_save_substate_FILE3D(fstream& f, const char* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);


template int CA_load_substate_FILE3D(const char* const path, double* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template int CA_load_substate_FILE3D(const char* const path, bool* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template int CA_load_substate_FILE3D(const char* const path, float* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template int CA_load_substate_FILE3D(const char* const path, int* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template int CA_load_substate_FILE3D(const char* const path, char* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);


template int CA_save_substate_FILE3D(const char* const path, const bool* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template int CA_save_substate_FILE3D(const char* const path, const double* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template int CA_save_substate_FILE3D(const char* const path, const float* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template int CA_save_substate_FILE3D(const char* const path, const int* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template int CA_save_substate_FILE3D(const char* const path, const char* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);


template void CA_print_STDOUT3D(const double* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template void CA_print_STDOUT3D(const bool* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template void CA_print_STDOUT3D(const float* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template void CA_print_STDOUT3D(const int* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
template void CA_print_STDOUT3D(const char* const Q,unsigned int yDim,unsigned int xDim, unsigned int zDim);
