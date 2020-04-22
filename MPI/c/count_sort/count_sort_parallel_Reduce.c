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
void count_sort(int a[], int n, int start, int stop);
void display_time(double start, double end);

int main(int argc, char *argv[]){ 

    int rank, size;
    double begin, end;
    int init_array[N], sorted_array[N];

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == ROOT) {
        rand_init_array(init_array, N, UPPER, LOWER);
        (void) printf("Initial array: ");
        display_array(init_array, N);
        (void) printf("Sorting began...\n\n");
    }

    MPI_Bcast(init_array, N, MPI_INT, ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) begin = MPI_Wtime(); /*  if rank is 0, start counting    */

    /* These initialization will be done by all processes   */
    int chunk = N / size;
    int extra = N % size;
    int start = rank * chunk;
    int stop = start + chunk;

    if (rank == size - 1) stop += extra;
    
    count_sort(init_array, N, start, stop);

    MPI_Reduce(init_array, sorted_array, N, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);

    if (rank == ROOT){
        end = MPI_Wtime();
        (void) printf("\nSorted array: ");
        display_array(sorted_array, N);
        display_time(begin, end);
    }

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

void count_sort(int a[], int n, int start, int stop) {
    int i, j, count;
    int* temp = calloc(n, sizeof(int));
    for (i = start; i < stop; ++i) {
        count = 0;
        for (j = 0; j < n; ++j)
            if (a[j] < a[i])
                ++count;
            else if (a[j] == a[i] && j < i)
                ++count;
        temp[count] = a[i];
    }
    
    memcpy(a, temp, n*sizeof(int));
    free(temp);
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
