#include <stdio.h>
#include <malloc.h>
#include "mpi.h"

#define N 21
#define ROOT 0

void init_array(int* a, int n);
void sieve(int* a, int n, int* loc, int start, int stop);
void attach_results(int* a, int* loc, int n);
void print_primes(int array[], int n);
void display_time(double start, double end);
 
int main(int argc, char *argv[])
{
    int rank, size;
    int* array;
    double begin, end;

    MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    array = (int*)calloc(sizeof(int), N+1);
    if (array == NULL) {printf ("Memory error\n"); MPI_Finalize(); return 4;}

    /* Initialize array with ones, except positions 0 and 1 */
    if (rank == ROOT)
        init_array(array, N+1);

    /* Broadcast initialized array to all processes */
    MPI_Bcast(array, N+1, MPI_INT, ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) begin = MPI_Wtime(); /*  if rank is 0, start counting */

    /* These initialization will be done by all processes   */
    int chunk = N / size;
    int extra = N % size;
    int start = (rank * chunk)+1;
    int stop = start + chunk;

    if (rank == size - 1) stop += extra; /* the last process takes the extra */
    else if (rank == ROOT) ++start; /* start from 2 */

    /* Each thread will have different locations vector */
    int* locations = (int*)malloc(sizeof(int)*(N+1));
    if (locations == NULL) {printf ("Memory error\n"); MPI_Finalize(); return 4;}

    /* Perform sieve */
    sieve(array, N+1, locations, start, stop);

    if (rank == ROOT){
        if (rank == ROOT) end = MPI_Wtime(); /*  if rank is 0, stop counting */
        (void) printf("\nPrimes numbers from 1 to %d are : ", N+1);
        print_primes(array, N+1);
        display_time(begin, end);
    }
    
    MPI_Finalize();

    return 0;
}

/*
 * Function:  init_array 
 * --------------------
 * Fills an integer array with ones, except the first two positions
 *
 *  array: the array that will be filled with numbers
 *  n: number of elements in the array
 *
 */

void init_array(int* a, int n){
    int i;
    for(i=2; i<n; ++i)
        a[i] = 1;
}

/*
 * Function:  sieve 
 * --------------------
 * Performs sieve
 *
 *  a: pointer of the array that contains ones
 *  n: number of elements in the array
 *  loc: pointer of the array that will contain positions of the non-prime numbers
 *  start: position from which the process will start
 *  stop: position that the process will stop
 *
 */
 
void sieve(int* a, int n, int* loc, int start, int stop){
    int i, j, k=0;
    for(i=start; i<=stop; ++i){
        if(a[i] == 1){
            for(j=i; (i*j)<n; ++j){
                loc[k] = i*j;
                ++k;
            }
        }
    }

    /* attach results to a (global array) */
    attach_results(a, loc, k);

}

/*
 * Function:  attach_results 
 * --------------------
 * Attaches results to global array a, after tracing locations of non prime numbers
 *
 *  a: pointer of the array that contains ones
 *  loc: pointer of the array that will contain positions of the non-prime numbers
 *  n: loc's number of elements
 *
 */

void attach_results(int* a, int* loc, int n){
    int i;
    for(i=0; i<n; ++i)
        a[loc[i]] = 0;
}

/*
 * Function:  print_primes 
 * --------------------
 * Prints prime numbers (position is prime when element of position is 1)
 *
 *  array: the array that contains 0 and 1 (0 if position is not prime, 1 if it is)
 *  n: number of elements in the array
 *
 */

void print_primes(int array[], int n){
    (void) printf("[ ");
    int i;
    for (i=0; i<n; ++i)
        if (array[i])
            (void) printf("%d ", i);
    (void) printf("]\n\n");
}

/*
 * Function:  display_time 
 * --------------------
 * Prints the time (seconds) elapsed
 *
 *  start: the time (seconds) in which the whole process started
 *  end: the time (seconds) in which the whole process started
 *
 */

void display_time(double start, double end){
    (void) printf("Time spent for sorting: %g seconds\n\n", (double)(end-start));
}