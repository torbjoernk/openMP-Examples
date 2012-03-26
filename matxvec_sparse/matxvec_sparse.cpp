#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"

#define N_COL 5
#define N_ROW 4
#define NNZ 7

void sparse2full( double *full, double *val, int *colInd, int *rowPt );
void printFullMat( double *full );
void printSparseMat ( double val[], int colInd[], int rowPt[] );
void printVec ( double val[], int size );
void mxv ( double * __restrict__ aval, int * __restrict__ acolind, int * __restrict__ arowpt,
           double * __restrict__ vval, double * __restrict__ yval );

/**
 * 
 */
int main( int argn, char *arg[] )
{
  srand( 1234567 );
  omp_set_num_threads(2);
  
  printf( "Number columns:      %u\n", N_COL );
  printf( "Number rows:         %u\n", N_ROW );
  printf( "Total Number Values: %u\n", N_ROW * N_COL );
  printf( "Number Non-Zeros:    %u\n", NNZ );
  printf( "Max Number Threads:  %u\n", omp_get_max_threads() );
  
  // allocate some memory
  // ... for matrix and fill it with random values
  double Aval[NNZ] = {1.0, 3.0, 4.0, 2.0, 5.0, 2.0, 1.0};
  int AcolInd[NNZ] = {1,   3,   2,   0,   4,   2,   3};
  int ArowPt[N_ROW+1] = {0, 2, 3, 5, 6};
  double Full[N_ROW*N_COL];
  sparse2full(Full, Aval, AcolInd, ArowPt);
  
  // ... for vector and fill it with random, non-zero, values
  double Vval[N_COL] = {1.0, 3.0, 4.0, 2.0, 3.0};
  
  // ... for result and make sure, it's zero everywhere
  double Yval[N_ROW];
  for ( int i = 0; i < N_ROW; i++ ) {
    Yval[i] = 0;
  }
  
  // print input if dimenions not too high
  if ( N_COL < 10 && N_ROW < 10 && NNZ < 15 ) {
    printSparseMat( Aval, AcolInd, ArowPt );
    printFullMat( Full );
    printVec( Vval, N_COL );
  }
  
  // multiply --- here we go!
  mxv( Aval, AcolInd, ArowPt, Vval, Yval );
  
  // print result if dimenions not too high
  if ( N_ROW < 10 ) {
    printVec( Yval, N_ROW );
  }
  // print squared norm of solution vector as a measurement for correctness
  double sqnorm = 0;
  for ( int i = 0; i < N_ROW; i++ ) {
    sqnorm += Yval[i] * Yval[i];
  }
  printf( "Squared Norm of Y is: % 10.2f\n", sqnorm );
}

void mxv(double * __restrict__ aval,
         int * __restrict__ acolind,
         int * __restrict__ arowpt,
         double * __restrict__ vval,
         double * __restrict__ yval)
{
  printf( "Multiplying ...\n" );
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
//       printf( "Thread %u is doing row=%u (x=%u)\tarowpt[x]=%u\tarowpt[x+1]=%u\n", omp_get_thread_num(), x+1, x, arowpt[x], arowpt[x+1] );
      for ( y = arowpt[x]; y < arowpt[x+1]; y++ ) {
//         printf( "\t[x,y]=[%u,%u]: aval[y]=% 4.2f\tacolind[y]=%u\tvval[acolind[y]]=% 4.2f\n", x, y, aval[y], acolind[y], vval[acolind[y]] );
        yval[x] += aval[y] * vval[ acolind[y] ];
      }
    }
  } /* end PARALLEL */
  printf( "... done.\n" );
}

/**
 * 
 */
void sparse2full( double *full, double *val, int *colInd, int *rowPt )
{
  for ( int row = 0; row < N_ROW; row++ ) {
    // fill everything with zeros
    for ( int col = 0; col < N_COL; col++ ) {
      full[ (row * N_COL) + col ] = 0;
    }
    
//     printf( "row=%u\tvalPt=[%u,%u)\n", row , rowPt[row], ((row+1 < N_ROW) ? rowPt[row+1] : NNZ ));
    for ( int valPt = rowPt[row]; valPt < ((row+1 < N_ROW) ? rowPt[row+1] : NNZ ); valPt++ ) {
//       printf( "\tvalPt=%u\tcolInd=%u\tval=% 4.2f\n", valPt, colInd[valPt], val[valPt] );
      full[ (row * N_COL) + colInd[valPt] ] = val[ valPt ];
    }
  }
}

/**
 * 
 */
void printFullMat( double *full )
{
  printf( "Full Matrix:\n" );
  for ( int row = 0; row < N_ROW; row++ ) {
    for ( int col = 0; col < N_COL; col++ ) {
      printf( "\t% 4.2f", full[ (row * N_COL) + col ] );
    }
    printf( "\n" );
  }
}

/**
 * 
 */
void printSparseMat( double *val, int *colInd, int *rowPt )
{
  printf( "Sparse Matrix in CRS format:\n" );
  printf( "\tValues:\t" );
  for ( int i = 0; i < NNZ; i++ ) {
    printf( "% 4.2f\t", val[i] );
  }
  
  printf( "\n\tColInd:\t" );
  for ( int i = 0; i < NNZ; i++ ) {
    printf( "%u\t", colInd[i] );
  }
  
  printf( "\n\tRowpt: \t" );
  for ( int i = 0; i < N_ROW+1; i++ ) {
    printf( "%u\t", rowPt[i] );
  }
  printf( "\n" );
}

/**
 * 
 */
void printVec( double *val, int size )
{
  printf( "Vector:\b" );
  for ( int i = 0; i < size; i++ ) {
    printf( "\t% 4.2f", val[i] );
  }
  printf( "\n" );
}
