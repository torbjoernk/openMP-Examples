#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <boost/concept_check.hpp>
#include "omp.h"

// define defaults:
int default_n_col = 5;
int default_n_row = 4;
int default_nnz = 7;
int default_n_threads = 2;
int default_seed = 1234567;

int *n_col = &default_n_col;
int *n_row = &default_n_row;
int *nnz = &default_nnz;
int *n_threads = &default_n_threads;
int *seed = &default_seed;

void full2sparse( double *full, double *val, int *colInd, int *rowPt );
void sparse2full( double *full, double *val, int *colInd, int *rowPt );
void printFullMat( double *full );
void printSparseMat( double *val, int *colInd, int *rowPt );
void printVec( double *val, int size );
void mxv( double * __restrict__ aval, int * __restrict__ acolind, int * __restrict__ arowpt,
          double * __restrict__ vval, double * __restrict__ yval );
void test_sparse();
void test_sparse_with_full();

/**
 * 
 */
int main( int argn, char *args[] )
{
  switch (argn) {
    case 6:
      *seed = atoi(args[5]);
    case 5:
      *n_threads = atoi(args[4]);
    case 4:
      *nnz = atoi(args[3]);
    case 3:
      *n_row = atoi(args[2]);
    case 2:
      *n_col = atoi(args[1]);
    default:
      break;
  }
  
  srand( *seed );
  omp_set_num_threads( *n_threads );
  
  printf( "Number columns:      %u\n", *n_col );
  printf( "Number rows:         %u\n", *n_row );
  printf( "Total Number Values: %u\n", *n_row * *n_col );
  printf( "Number Non-Zeros:    %u\n", *nnz );
  printf( "Max Number Threads:  %u\n", omp_get_max_threads() );
  printf( "Random Seed:         %u\n", *seed );

  // this test uses hard-coded test values
  if ( *nnz == 7 && *n_row == 4 && *n_col == 5 ) {
    test_sparse();
  }
  
  // this test is variable
  test_sparse_with_full();
  
  return(0);
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
    shared(n_row, aval, acolind, arowpt, vval, yval) \
    private(x, y)
  {
    #pragma omp for \
      schedule(static)
    for ( x = 0; x < *n_row; x++ ) {
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
  for ( int row = 0; row < *n_row; row++ ) {
    // fill everything with zeros
    for ( int col = 0; col < *n_col; col++ ) {
      full[ (row * *n_col) + col ] = 0;
    }
    
//     printf( "row=%u\tvalPt=[%u,%u)\n", row , rowPt[row], rowPt[row+1] );
    for ( int valPt = rowPt[row]; valPt < rowPt[row+1]; valPt++ ) {
//       printf( "\tvalPt=%u\tcolInd=%u\tval=% 4.2f\n", valPt, colInd[valPt], val[valPt] );
      full[ (row * *n_col) + colInd[valPt] ] = val[ valPt ];
    }
  }
}

/**
 * 
 */
void full2sparse( double *full, double *val, int *colInd, int *rowPt )
{
  int curr_nnz = 0;
  for ( int row = 0; row < *n_row; row++ ) {
    rowPt[row] = curr_nnz;
    for ( int col = 0; col < *n_col; col++ ) {
      if ( full[ (row * *n_col) + col ] != 0.0 ) {
        val[curr_nnz] = full[ (row * *n_col) + col ];
        colInd[curr_nnz] = col;
        curr_nnz++;
      }
    }
  }
  rowPt[*n_row] = *nnz;
}

/**
 * 
 */
void printFullMat( double *full )
{
  printf( "Full Matrix:\n" );
  for ( int row = 0; row < *n_row; row++ ) {
    for ( int col = 0; col < *n_col; col++ ) {
      printf( "\t% 4.2f", full[ (row * *n_col) + col ] );
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
  for ( int i = 0; i < *nnz; i++ ) {
    printf( "% 4.2f\t", val[i] );
  }
  
  printf( "\n\tColInd:\t" );
  for ( int i = 0; i < *nnz; i++ ) {
    printf( "%u\t", colInd[i] );
  }
  
  printf( "\n\tRowpt: \t" );
  for ( int i = 0; i < (*n_row)+1; i++ ) {
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
  double *Aval = new double[7];
  Aval[0] = 1.0;
  Aval[1] = 3.0;
  Aval[2] = 4.0;
  Aval[3] = 2.0;
  Aval[4] = 5.0;
  Aval[5] = 2.0;
  Aval[6] = 1.0;
  int *AcolInd = new int[7];
  AcolInd[0] = 1;
  AcolInd[1] = 3;
  AcolInd[2] = 2;
  AcolInd[3] = 0;
  AcolInd[4] = 4;
  AcolInd[5] = 2;
  AcolInd[6] = 3;
  int *ArowPt = new int[4+1];
  ArowPt[0] = 0;
  ArowPt[1] = 2;
  ArowPt[2] = 3;
  ArowPt[3] = 5;
  ArowPt[4] = 7;
  double *Full = new double[4*5];
  sparse2full(Full, Aval, AcolInd, ArowPt);
  
  // ... for vector and fill it with random, non-zero, values
  double *Vval = new double[5];
  Vval[0] = 1.0;
  Vval[1] = 3.0;
  Vval[2] = 4.0;
  Vval[3] = 2.0;
  Vval[4] = 3.0;
  
  // ... for result and make sure, it's zero everywhere
  double *Yval = new double[4];
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
  
  delete[] Aval;
  Aval = NULL;
  delete[] AcolInd;
  AcolInd = NULL;
  delete[] ArowPt;
  ArowPt = NULL;
  delete[] Full;
  Full = NULL;
  delete[] Vval;
  Vval = NULL;
  delete[] Yval;
  Yval = NULL;
}

/**
 * 
 */
void test_sparse_with_full()
{
  printf( "\n*** Testing random full ...\n" );
  // allocate memory
  double *Full = new double[*n_row * *n_col];
  double *Aval = new double[*nnz];
  int *AcolInd = new int[*nnz];
  int *ArowPt = new int[(*n_row)+1];
  double *Vval = new double[*n_col];
  double *Yval = new double[*n_row];
  
  if ( nnz < n_row ) {
    printf( "ERROR: n_row must not be bigger than nnz. Singularity matrix otherwise. (nnz=%u, n_row=%u)\n", nnz, n_row );
    exit( -1 );
  }
  
  // initialize full matrix
  for ( int i = 0; i < *n_col * *n_row; i++ ) {
    Full[i] = 0.0;
  }
  
  // fill full matrix with random values
  // ... first make sure, one value per row
  for ( int row = 0; row < *n_row; row++ ) {
    int col = rand() % *n_col;
    Full[ row * *n_col + col ] = rand() % 5 + 1;
  }
  // then the remaining
  for ( int remaining_nnz = *nnz - *n_row; remaining_nnz > 0; remaining_nnz-- ) {
    int row = 0, col = 0;
    bool already_nnz = true;
    while ( already_nnz ) {
      row = rand() % *n_row;
      col = rand() % *n_col;
      already_nnz = ( Full[ row * *n_col + col ] != 0 );
    }
    Full[ row * *n_col + col ] = rand() % 5 + 1;
  }
  // convert the full to a sparse
  full2sparse( Full, Aval, AcolInd, ArowPt );
  
  // fill the vector randomly
  for ( int i = 0; i < *n_col; i++ ) {
    Vval[i] = rand() % 5 + 1;
  }
  
  // for result vector make sure, it's zero everywhere
  for ( int i = 0; i < *n_row; i++ ) {
    Yval[i] = 0;
  }
  
  // print input if dimenions not too high
  if ( *n_col < 10 && *n_row < 10 && *nnz < 15 ) {
    printFullMat( Full );
    printSparseMat( Aval, AcolInd, ArowPt );
    printVec( Vval, *n_col );
  }
  
  // multiply --- here we go!
  mxv( Aval, AcolInd, ArowPt, Vval, Yval );
  
  // print result if dimenions not too high
  if ( *n_row < 10 ) {
    printVec( Yval, *n_row );
  }
  // print squared norm of solution vector as a measurement for correctness
  double sqnorm = 0;
  for ( int i = 0; i < *n_row; i++ ) {
    sqnorm += Yval[i] * Yval[i];
  }
  printf( "Squared Norm of Y is: % 10.2f\n", sqnorm );
  
  // free memory
  delete[] Full;
  Full = NULL;
  delete[] Aval;
  Aval = NULL;
  delete[] AcolInd;
  AcolInd = NULL;
  delete[] ArowPt;
  ArowPt = NULL;
  delete[] Vval;
  Vval = NULL;
  delete[] Yval;
  Yval = NULL;
}
