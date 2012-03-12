#include <time.h>
#include <stdio.h>
#include <omp.h>

#define NTHREADS 2
#define N 100000
#define CHUNK 10

int main( int argn, char *arg[] )
{
  omp_set_num_threads( NTHREADS );
  
  printf( "Number threads: %u\n", NTHREADS );
  
  timespec start1, start2;
  start1.tv_sec = 0;
  start1.tv_nsec = 0;
  start2.tv_sec = 0;
  start2.tv_nsec = 0;
  double sum;
  double a[N], b[N];
  int i;
  
  clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &start1 );
  for ( i = 0; i < N; i++ ) {
    a[i] = i * 0.5;
    b[i] = i * 2.0;
  }
  clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &start1 );
  
  sum = 0;
  
  clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &start2 );
  #pragma omp parallel for \
    default(shared) \
    shared(a, b) \
    reduction(+:sum) \
    schedule(static, CHUNK)
  for ( i = 0; i < N; i++ ) {
    sum = sum + a[i] * b[i];
  }
  clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &start2 );
  
  printf( "sum = %f\n", sum );
  printf( "Initialization\n%u sec % 10u nsec\n",
          start1.tv_sec, start1.tv_nsec );
  printf( "Addition\n%u sec % 10u nsec\n",
          start2.tv_sec, start2.tv_nsec );
}
