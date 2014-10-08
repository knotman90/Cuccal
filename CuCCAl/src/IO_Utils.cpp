/*
 * IO_Utils.cpp
 *
 *  Created on: 21/mar/2014
 *      Author: Donato D'ambrosio, Davide Spataro
 *
 */


#include "IO_Utils.h"
using namespace std;

template <typename type>
void CA_print_STDOUT(const type* const Q,unsigned int rows,unsigned int cols){
	cout<<"START-----------------"<<endl;
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			cout<<Q[i*cols+j]<<" ";
		}
		cout<<endl;
	}
	cout<<"END-----------------"<<endl;
}


/* -----------save/load -> to/from FILES  */
template <class type>
void CA_load_substate_FILE(fstream& f, type* const Q,unsigned int rows,unsigned int cols)
{

	for (int i=0; i<rows; i++){
		for (int j=0; j<cols; j++){
			f >> Q[getLinearIndexNormal2D(i,j,rows,cols)];
		}
	}

}



template <class type>
int CA_load_substate_FILE(const char* const path, type* const Q,unsigned int rows,unsigned int cols)
{
	fstream f ( path, ios::in );
	if ( !f.is_open() )
		return ERROR_OPENING_FILE;

	CA_load_substate_FILE(f, Q,rows,cols);

	f.close();

	return SUCCESS_OPENING_FILE;
}

template <class type>
void CA_save_substate_FILE(fstream& f, const type* const Q,unsigned int rows,unsigned int cols)
{

	for (int i=0; i<rows; i++) {
		for (int j=0; j<cols; j++) {
			f << Q[getLinearIndexNormal2D(i,j,rows,cols)] << " ";
		}
		f << "\n";
	}
}


template <class type>
int CA_save_substate_FILE(const char* const path, const type* const Q,unsigned int rows,unsigned int cols)
{
	fstream f ( path, ios::out );
	if ( !f.is_open() )
		return ERROR_OPENING_FILE;

	CA_save_substate_FILE(f, Q,rows,cols);

	f.close();

	return SUCCESS_OPENING_FILE;
}

/* END -----------save/load -> to/from FILES*/





//-----------------------------------------------------------------------
//DEFINITION OF THE TEMPLATE FUNCTIONS
//-----------------------------------------------------------------------

template void CA_load_substate_FILE(fstream& f, bool* const Q,unsigned int rows,unsigned int cols);
template void CA_load_substate_FILE(fstream& f, double* const Q,unsigned int rows,unsigned int cols);
template void CA_load_substate_FILE(fstream& f, float* const Q,unsigned int rows,unsigned int cols);
template void CA_load_substate_FILE(fstream& f, int* const Q,unsigned int rows,unsigned int cols);
template void CA_load_substate_FILE(fstream& f, char* const Q,unsigned int rows,unsigned int cols);

template void CA_save_substate_FILE(fstream& f, const bool* const Q,unsigned int rows,unsigned int cols);
template void CA_save_substate_FILE(fstream& f, const double* const Q,unsigned int rows,unsigned int cols);
template void CA_save_substate_FILE(fstream& f, const float* const Q,unsigned int rows,unsigned int cols);
template void CA_save_substate_FILE(fstream& f, const int* const Q,unsigned int rows,unsigned int cols);
template void CA_save_substate_FILE(fstream& f, const char* const Q,unsigned int rows,unsigned int cols);


template int CA_load_substate_FILE(const char* const path, double* const Q,unsigned int rows,unsigned int cols);
template int CA_load_substate_FILE(const char* const path, bool* const Q,unsigned int rows,unsigned int cols);
template int CA_load_substate_FILE(const char* const path, float* const Q,unsigned int rows,unsigned int cols);
template int CA_load_substate_FILE(const char* const path, int* const Q,unsigned int rows,unsigned int cols);
template int CA_load_substate_FILE(const char* const path, char* const Q,unsigned int rows,unsigned int cols);


template int CA_save_substate_FILE(const char* const path, const bool* const Q,unsigned int rows,unsigned int cols);
template int CA_save_substate_FILE(const char* const path, const double* const Q,unsigned int rows,unsigned int cols);
template int CA_save_substate_FILE(const char* const path, const float* const Q,unsigned int rows,unsigned int cols);
template int CA_save_substate_FILE(const char* const path, const int* const Q,unsigned int rows,unsigned int cols);
template int CA_save_substate_FILE(const char* const path, const char* const Q,unsigned int rows,unsigned int cols);



template void CA_print_STDOUT(const double* const Q,unsigned int rows,unsigned int cols);
template void CA_print_STDOUT(const bool* const Q,unsigned int rows,unsigned int cols);
template void CA_print_STDOUT(const float* const Q,unsigned int rows,unsigned int cols);
template void CA_print_STDOUT(const int* const Q,unsigned int rows,unsigned int cols);
template void CA_print_STDOUT(const char* const Q,unsigned int rows,unsigned int cols);




