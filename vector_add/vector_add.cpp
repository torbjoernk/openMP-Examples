#include <time.h>
#include <stdio.h>
#include <omp.h>

#define NTHREADS 2
#define N 1000000
#define CHUNKSIZE 1

main()
{
  omp_set_num_threads( NTHREADS );
  
  for ( int n = 0; n < 1; n++ ) {
    timespec start1, start2;
    start1.tv_sec = 0;
    start1.tv_nsec = 0;
    start2.tv_sec = 0;
    start2.tv_nsec = 0;
    int nthreads, tid;
    int arr1[N], arr2[N], arr3[N];
    
    printf( "int\tuint\tappr.Stack\tNthreads\tchunk\n" );
    printf( "%u\t%u\t%u\t%u\t\t%u\n",
            sizeof( int ), sizeof( unsigned int ), sizeof( int ) * N * 3,
            omp_get_max_threads(), CHUNKSIZE );
    
    // initialize arrays
    clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &start1 );
    #pragma omp parallel shared(arr1, arr2)
    {
      #pragma omp for schedule(dynamic, CHUNKSIZE)
      for ( int i = 0; i < N; i++ ) {
        arr1[i] = i;
        arr2[i] = i;
      }
    } // END parallel shared(arr1, arr2)
    clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &start1 );
    
    // add arrays
    clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &start2 );
    #pragma omp parallel shared(arr1, arr2, arr3)
    {
      #pragma omp for schedule(dynamic, CHUNKSIZE)
      for ( int i = 0; i < N; i++ ) {
//         arr3[i] = arr1[i] + arr2[i];
        arr3[i] = arr1[i] * 2 + arr2[i] * 3;
      }
    } // END parallel shared(arr1, arr2, arr3)
    clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &start2 );
    
    printf( "Initialization\t\tAddition\n" );
    printf( "%u sec % 10u nsec\t%u sec % 10u nsec\n",
            start1.tv_sec, start1.tv_nsec, start2.tv_sec, start2.tv_nsec );
  }
}
