#include <iostream>
#include <stdlib.h>
#include <time.h>
#include "omp.h"

#define N_COL 5
#define N_ROW 4
#define NNZ 6

void printSparseMat ( double val[], int colInd[], int rowPt[] );
void printVec ( double val[], int size );
void mxv ( double * __restrict__ aval, int * __restrict__ acolind, int * __restrict__ arowpt,
           double * __restrict__ vval, double * __restrict__ yval );

/**
 * 
 */
int main( int argn, char *arg[] )
{
  srand( 123 );
  omp_set_num_threads(4);
  
  std::cout << "Number columns:      " << N_COL << std::endl;
  std::cout << "Number rows:         " << N_ROW << std::endl;
  std::cout << "Total Number Values: " << N_ROW * N_COL << std::endl;
  std::cout << "Number Non-Zeros:    " << NNZ << std::endl;
  std::cout << "Max Number Threads:  " << omp_get_max_threads() << std::endl;
  
  // allocate some memory
  // ... for matrix and fill it with random values
  double Aval[NNZ];
  int AcolInd[NNZ];
  for ( int i = 0; i < NNZ; i++ ) {
    Aval[i] = rand() % 5 + 1;
    AcolInd[i] = rand() % N_COL;
  }
  int ArowPt[N_ROW];
  ArowPt[0] = 0;
  for ( int i = 1; i < N_ROW; i++ ) {
    ArowPt[i] = rand() % (NNZ - ArowPt[i-1] - (N_ROW - i)) + ArowPt[i-1] + 1;
  }
  
  // ... for vector and fill it with random, non-zero, values
  double Vval[N_COL];
  for ( int i = 0; i < N_COL; i++ ) {
    Vval[i] = rand() % 5 + 1;
  }
  
  // ... for result and make sure, it's zero everywhere
  double Yval[N_ROW];
  for ( int i = 0; i < N_ROW; i++ ) {
    Yval[i] = 0;
  }
  
  // print input if dimenions not too high
  if ( N_COL < 10 && N_ROW < 10 && NNZ < 15 ) {
    printSparseMat( Aval, AcolInd, ArowPt );
    printVec( Vval, N_COL );
  }
  
  // multiply --- here we go!
  mxv( Aval, AcolInd, ArowPt, Vval, Yval );
  
  // print result if dimenions not too high
  if ( N_COL < 10 && N_ROW < 10 && NNZ < 15 ) {
    printVec( Yval, N_ROW );
  }
  // print squared norm of solution vector as a measurement for correctness
  double sqnorm = 0;
  for ( int i = 0; i < N_ROW; i++ ) {
    sqnorm += Yval[i] * Yval[i];
  }
  std::cout << "Squared Norm of Y is: " << sqnorm << std::endl;
}

void mxv(double * __restrict__ aval,
         int * __restrict__ acolind,
         int * __restrict__ arowpt,
         double * __restrict__ vval,
         double * __restrict__ yval)
{
  int x, y = 0;
  #pragma omp parallel \
    default(none) \
    shared(aval, acolind, arowpt, vval, yval) \
    private(x, y)
  {
    #pragma omp for \
      schedule(static)
    for ( x = 0; x < N_ROW; x++ ) {
      yval[x] = 0;
      for ( y = arowpt[x]; y < arowpt[x+1]; y++ ) {
        yval[x] += aval[y] * vval[ acolind[y] ];
      }
    }
  } /* end PARALLEL */
}

/**
 * 
 */
void printSparseMat( double val[], int colInd[], int rowPt[] )
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
  for ( int i = 0; i < N_ROW; i++ ) {
    std::cout << rowPt[i] << "\t";
  }
  std::cout << std::endl;
}

/**
 * 
 */
void printVec( double val[], int size )
{
  std::cout << "Vector:" << std::endl;
  for ( int i = 0; i < size; i++ ) {
    std::cout << "\t" << val[i];
  }
  std::cout << std::endl;
}
