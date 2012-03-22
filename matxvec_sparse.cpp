#include <iostream>
#include <stdlib.h>
#include <time.h>
#include "omp.h"

#define N_COL 5000
#define N_ROW 4000
#define NNZ 6000

void printSparseMat ( double val[], int colInd[], int rowPt[] );
void printVec ( double val[], int size );

int main( int argn, char *arg[] )
{
  srand( 123 );
  omp_set_num_threads(4);
  
  std::cout << "Number columns:      " << N_COL << std::endl;
  std::cout << "Number rows:         " << N_ROW << std::endl;
  std::cout << "Total Number Values: " << N_ROW * N_COL << std::endl;
  std::cout << "Number Non-Zeros:    " << NNZ << std::endl;
  std::cout << "Max Number Threads:  " << omp_get_max_threads() << std::endl;
  
  // allocate memory
  // ... for matrix
  double Aval[NNZ];
  int AcolInd[NNZ];
  for ( int i = 0; i < NNZ; i++ ) {
    Aval[i] = rand() % 50 + 1;
    AcolInd[i] = rand() % N_COL;
  }
  int ArowPt[N_ROW];
  ArowPt[0] = 0;
  for ( int i = 1; i < NNZ; i++ ) {
    ArowPt[i] = rand() % (NNZ - ArowPt[i-1] - (N_ROW - i)) + ArowPt[i-1] + 1;
  }
  
  // ... for vector
  double Vval[N_COL];
  for ( int i = 0; i < N_COL; i++ ) {
    Vval[i] = rand() % 50 + 1;
  }
  
  // ... for result
  double Yval[N_ROW];
  
  // print input
  if ( N_COL < 10 && N_ROW < 10 && NNZ < 15 ) {
    printSparseMat( Aval, AcolInd, ArowPt );
    printVec( Vval, N_COL );
  }
  
  // multiply
  int x, y = 0;
#pragma omp parallel for \
  default(none) \
  shared(Aval, AcolInd, ArowPt, Vval, Yval) \
  private(x, y) \
  schedule(static)
  for ( x = 0; x < N_ROW; x++ ) {
    Yval[x] = 0;
    for ( y = ArowPt[x]; y < ArowPt[x+1]; y++ ) {
      Yval[x] += Aval[y] * Vval[ AcolInd[y]-1 ];
    }
  }
  
  // print result
  if ( N_COL < 10 && N_ROW < 10 && NNZ < 15 ) {
    printVec( Yval, N_ROW );
  } else {
    double sqnorm = 0;
    for ( int i = 0; i < N_ROW; i++ ) {
      sqnorm += Yval[i] * Yval[i];
    }
    std::cout << "Squared Norm of Y is: " << sqnorm << std::endl;
  }
}

void printSparseMat( double val[], double colInd[], double rowPt[] )
{
  std::cout << "Sparse Matrix in CRS format:" << std::endl;
  std::cout << "\tValues:\t";
  for ( int i = 0; i < NNZ; i++ ) {
    std::cout << val[i] << "\t";
  }
  
  std::cout << std::endl << "\tColInd:\t";
  for ( int i = 0; i < NNZ; i++ ) {
    std::cout << colInd[i] << "\t";
  }
  
  std::cout << std::endl << "\tRowpt: \t";
  for ( int i = 0; i < N_COL + 1; i++ ) {
    std::cout << rowPt[i] << "\t";
  }
  std::cout << std::endl;
}

void printVec( double val[], int size )
{
  std::cout << "Vector:" << std::endl;
  for ( int i = 0; i < size; i++ ) {
    std::cout << "\t" << val[i];
  }
  std::cout << std::endl;
}
