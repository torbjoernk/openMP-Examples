#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <boost/concept_check.hpp>
#include "omp.h"

#define N_COL 50
#define N_ROW 40
#define NNZ 60

void full2sparse( double *full, double *val, int *colInd, int *rowPt );
void sparse2full( double *full, double *val, int *colInd, int *rowPt );
void printFullMat( double *full );
void printSparseMat( double val[], int colInd[], int rowPt[] );
void printVec( double val[], int size );
void mxv( double * __restrict__ aval, int * __restrict__ acolind, int * __restrict__ arowpt,
          double * __restrict__ vval, double * __restrict__ yval );
void test_sparse();
void test_sparse_with_full();

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

  // this test uses hard-coded test values
  if ( NNZ == 7 && N_ROW == 4 && N_COL == 5 ) {
    test_sparse();
  }
  
  // this test is variable
  test_sparse_with_full();
}

void mxv(double * __restrict__ aval,
         int * __restrict__ acolind,
         int * __restrict__ arowpt,
         double * __restrict__ vval,
         double * __restrict__ yval)
{
  printf( "Multiplying ..." );
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
//       printf( "\nThread %u is doing row=%u (x=%u)\tarowpt[x]=%u\tarowpt[x+1]=%u", omp_get_thread_num(), x+1, x, arowpt[x], arowpt[x+1] );
      for ( y = arowpt[x]; y < arowpt[x+1]; y++ ) {
//         printf( "\n\t[x,y]=[%u,%u]: aval[y]=% 4.2f\tacolind[y]=%u\tvval[acolind[y]]=% 4.2f", x, y, aval[y], acolind[y], vval[acolind[y]] );
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
    
//     printf( "row=%u\tvalPt=[%u,%u)\n", row , rowPt[row], rowPt[row+1] );
    for ( int valPt = rowPt[row]; valPt < rowPt[row+1]; valPt++ ) {
//       printf( "\tvalPt=%u\tcolInd=%u\tval=% 4.2f\n", valPt, colInd[valPt], val[valPt] );
      full[ (row * N_COL) + colInd[valPt] ] = val[ valPt ];
    }
  }
}

/**
 * 
 */
void full2sparse( double *full, double *val, int *colInd, int *rowPt )
{
  int curr_nnz = 0;
  for ( int row = 0; row < N_ROW; row++ ) {
    rowPt[row] = curr_nnz;
    for ( int col = 0; col < N_COL; col++ ) {
      if ( full[ (row * N_COL) + col ] != 0.0 ) {
        val[curr_nnz] = full[ (row * N_COL) + col ];
        colInd[curr_nnz] = col;
        curr_nnz++;
      }
    }
  }
  rowPt[N_ROW] = NNZ;
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

/**
 * 
 */
void test_sparse()
{
  printf( "\n*** Testing default sparse ...\n" );
  // allocate some memory
  // ... for matrix and fill it with random values
  double Aval[7] = {1.0, 3.0, 4.0, 2.0, 5.0, 2.0, 1.0};
  int AcolInd[7] = {1,   3,   2,   0,   4,   2,   3};
  int ArowPt[4+1] = {0, 2, 3, 5, 7};
  double Full[4*5];
  sparse2full(Full, Aval, AcolInd, ArowPt);
  
  // ... for vector and fill it with random, non-zero, values
  double Vval[5] = {1.0, 3.0, 4.0, 2.0, 3.0};
  
  // ... for result and make sure, it's zero everywhere
  double Yval[4];
  for ( int i = 0; i < 4; i++ ) {
    Yval[i] = 0;
  }
  
  // print input
  printSparseMat( Aval, AcolInd, ArowPt );
  printFullMat( Full );
  printVec( Vval, 5 );
  
  // multiply --- here we go!
  mxv( Aval, AcolInd, ArowPt, Vval, Yval );
  
  // print result
  printVec( Yval, 4 );
  
  // print squared norm of solution vector as a measurement for correctness
  double sqnorm = 0;
  for ( int i = 0; i < 4; i++ ) {
    sqnorm += Yval[i] * Yval[i];
  }
  printf( "Squared Norm of Y is: % 10.2f\n", sqnorm );
}

/**
 * 
 */
void test_sparse_with_full()
{
  printf( "\n*** Testing random full ...\n" );
  // allocate memory
  double Full[N_ROW*N_COL];
  double Aval[NNZ];
  int AcolInd[NNZ];
  int ArowPt[N_ROW+1];
  double Vval[N_COL];
  double Yval[N_ROW];
  
  if ( NNZ < N_ROW ) {
    printf( "ERROR: N_ROW must be bigger than NNZ. Singularity matrix otherwise. (NNZ=%u, N_ROW=%u)\n", NNZ, N_ROW );
    exit( -1 );
  }
  
  // initialize full matrix
  for ( int i = 0; i < N_COL*N_ROW; i++ ) {
    Full[i] = 0.0;
  }
  
  // fill full matrix with random values
  // ... first make sure, one value per row
  for ( int row = 0; row < N_ROW; row++ ) {
    int col = rand() % N_COL;
    Full[ row*N_COL + col ] = rand() % 5 + 1;
  }
  // then the remaining
  for ( int remaining_nnz = NNZ - N_ROW; remaining_nnz >= 0; remaining_nnz-- ) {
    int row = 0, col = 0;
    bool already_nnz = true;
    while ( already_nnz ) {
      row = rand() % N_ROW;
      col = rand() % N_COL;
      already_nnz = ( Full[ row*N_COL + col ] != 0 );
    }
    Full[ row*N_COL + col ] = rand() % 5 + 1;
  }
  // convert the full to a sparse
  full2sparse( Full, Aval, AcolInd, ArowPt );
  
  // fill the vector randomly
  for ( int i = 0; i < N_COL; i++ ) {
    Vval[i] = rand() % 5 + 1;
  }
  
  // for result vector make sure, it's zero everywhere
  for ( int i = 0; i < N_ROW; i++ ) {
    Yval[i] = 0;
  }
  
  // print input if dimenions not too high
  if ( N_COL < 10 && N_ROW < 10 && NNZ < 15 ) {
    printFullMat( Full );
    printSparseMat( Aval, AcolInd, ArowPt );
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
