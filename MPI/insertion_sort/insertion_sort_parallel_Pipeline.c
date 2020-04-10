
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define tag 100  /* tag used in MPI_Send */

void rand_init_array(int array[], int n);
void insertion_sort(int array[], int n, int rank, int* smaller, int* received);
void fetch_results(int sorted_array[], int n, int rank, int* smaller);
void display_array(int array[], int n);

int main(int argc, char *argv[]){
    int rank, size;
    int received, smaller;
    MPI_Status status;
    int init_array[size], sorted_array[size];

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    
    if(!rank){
            rand_init_array(init_array, size);  /* Process 0: Initialize array */
            (void) printf("Initial array: ");
            display_array(init_array, size);
    }

    insertion_sort(init_array, size, rank, &smaller, &received);

    fetch_results(sorted_array, size, rank, &smaller);

    MPI_Finalize();

    return 0;
}

/*
 * Function:  insertion_sort 
 * --------------------
 * Sorts the array's elements (size in total)
 *
 *  array: the initial array
 *  n: number of elements in the array
 *  rank: the process that runs the function
 *  smaller: array's smallest element
 *  received: array's element that was last received
 *
 */

void insertion_sort(int array[], int n, int rank, int* smaller, int* received){
    MPI_Status status;
    int i;
    for(i=0; i<n-rank; ++i){

        if(!rank)
            *received = array[i];
        else
            MPI_Recv(received, 1, MPI_INT, rank-1, tag, MPI_COMM_WORLD, &status);

        if(!i)  /* if this is the first step */
            *smaller = *received; /* initializes smaller with received at step zero */
        else
            if(*received > *smaller)  /* compares smaller to received */
                MPI_Send(received, 1, MPI_INT, rank+1, tag, MPI_COMM_WORLD);   /* sends the larger number through */
        else{
            MPI_Send(smaller, 1, MPI_INT, rank+1, tag, MPI_COMM_WORLD);    /* sends the larger number through */
            *smaller = *received;
        }
    }
}

/*
 * Function:  fetch_results 
 * --------------------
 * Fetches elements from all processes (size in total)
 *
 *  sorted_array: the array with initial array's sorted elements 
 *  n: number of elements in the array
 *  rank: the process that runs the function
 *  smaller: array's smallest element
 *
 */

void fetch_results(int sorted_array[], int n, int rank, int* smaller){
    MPI_Status status;
    int i;
    if(!rank){  /* if is process 0 */
        sorted_array[0] = *smaller;
        for(i=1; i<n; ++i)
            MPI_Recv(&sorted_array[i], 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
        (void) printf("\nSorted array: ");
        display_array(sorted_array, n);
   }
   else
        MPI_Send(smaller, 1, MPI_INT, 0, tag, MPI_COMM_WORLD); /* send the smaller element to process 0 */
}

/*
 * Function:  rand_init_array 
 * --------------------
 * Fills an integer array with random numbers
 *
 *  array: the array that will be filled with numbers
 *  n: number of elements in the array
 *
 */

void rand_init_array(int array[], int n){
    int i;    
    for (i=0; i<n; ++i)
        array[i] = (rand() % n*i) + 1;
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
