#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

#define ROOT 0
#define N 8
#define UPPER N*4
#define LOWER 1

void rand_init_array(int array[], int n, int upper, int lower);
void display_array(int array[], int n);
void count_sort(int a[], int n, int start, int stop, int* locations);
void display_time(double start, double end);
void attatch_results(int buff[], int results[], int* locations, int n);

int main(int argc, char *argv[]){ 

    int rank, size;
    double begin, end;
    int init_array[N], sorted_array[N];
    int* locations;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == ROOT) {
        rand_init_array(init_array, N, UPPER, LOWER);
        (void) printf("Initial array: ");
        display_array(init_array, N);
        (void) printf("Sorting began...\n\n");

        int* locations = (int*)malloc(sizeof(int)*N);
        if (locations == NULL) {printf ("Memory error\n"); MPI_Finalize(); return 4;}

    }
    
    if (rank == ROOT) begin = MPI_Wtime(); /*  if rank is 0, start counting    */

    /* These initialization will be done by all processes   */
    int chunk = N / size;
    int start = rank * chunk;
    int stop = start + chunk;

    /* Each thread will have different local_locations vector */
    int* local_locations = (int*)malloc(sizeof(int)*(stop-start));
    if (local_locations == NULL) {printf ("Memory error\n"); MPI_Finalize(); return 4;}

    MPI_Bcast(init_array, N, MPI_INT, ROOT, MPI_COMM_WORLD);

    count_sort(init_array, N, start, stop, local_locations);

    if(rank == ROOT) {
        
        /* locations (global) will be the receive buffer, containing the right position of each element */
        locations = (int*)malloc(sizeof(int)*N);
        if (locations == NULL) {printf ("Memory error\n"); MPI_Finalize(); return 6;}
        
        MPI_Gather(local_locations, (stop-start), MPI_INT, locations, (stop-start), MPI_INT, ROOT, MPI_COMM_WORLD);

        attatch_results(init_array, sorted_array, locations, N);

        end = MPI_Wtime();

        (void) printf("\nSorted array: ");
        display_array(sorted_array, N);
        display_time(begin, end);
  
       free(locations);
    }
    else
        MPI_Gather(local_locations, (stop-start), MPI_INT, locations, (stop-start), MPI_INT, ROOT, MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;

}

/*
 * Function:  rand_init_array 
 * --------------------
 * Fills an integer array with random numbers
 *
 *  array: the array that will be filled with numbers
 *  n: number of elements in the array
 *  upper: highest value of random number
 *  lower: lowest value of random number
 *
 */

void rand_init_array(int array[], int n, int upper, int lower){
    int i;    
    for (i=0; i<n; ++i)
        array[i] = (rand() % (upper - lower + 1)) + lower;
}

/*
 * Function:  attatch_results 
 * --------------------
 * Attaches final results to results array, using a locations vector
 *
 *  buff: the initial array
 *  results: the results array, which will contain the elements of buff sorted
 *  locations: locations vector, contains the right positions for buff's elements (0...N, in order the elements to be sorted)
 *  n: number of elements in the array
 *
 */

void attatch_results(int buff[], int results[], int* locations, int n){
    int i;
    for(i=0; i<n; i++)
        results[locations[i]] = buff[i];
}

/*
 * Function:  display_array 
 * --------------------
 * Prints an integer array to user
 *
 *  array: the array that will be printed
 *  n: number of elements in the array
 *
 */

void display_array(int array[], int n){
    (void) printf("[ ");
    int i;
    for (i=0; i<n; ++i)
        (void) printf("%d ", array[i]);
    (void) printf("]\n\n");
}

/*
 * Function:  count_sort 
 * --------------------
 * Sorts an integer array, using the count sort algorithm (enumeration sort)
 *
 *  a: the array that will be sorted
 *  n: number of elements in the array
 *
 */

void count_sort(int a[], int n, int start, int stop, int* locations) {
    int i, j, count;
    for (i = start; i < stop; ++i) {
        count = 0;
        for (j = 0; j < n; ++j)
            if (a[j] < a[i])
                ++count;
            else if (a[j] == a[i] && j < i)
                ++count;
        locations[i - start] = count;
    }
}

/*
 * Function:  display_time 
 * --------------------
 * Prints the time (seconds) elapsed for sorting
 *
 *  start: the time (seconds) in which the sorting started
 *  end: the time (seconds) in which the sorting finished
 *
 */

void display_time(double start, double end){
    (void) printf("Time spent for sorting: %g seconds\n", (double)(end-start));
}
