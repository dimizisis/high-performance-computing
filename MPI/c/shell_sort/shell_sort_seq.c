
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 500000
#define ROOT 0
#define TAG 100 /* for MPI_Send */

void rand_init_array(int array[], int n, int upper, int lower);
void display_array(int array[], int n);
void shell_sort(int array[], int n);
 
int main(void) {
    int array[N];
    rand_init_array(array, N, 0, N*4);
    // (void) printf("Initial array: ");
    // display_array(array, N);

    clock_t begin = clock();

    shell_sort(array, N);

    clock_t end = clock();

    (void) printf("Time elapsed: %g\n", (double)(end-begin)/CLOCKS_PER_SEC);

    // (void) printf("Sorted array: ");
    // display_array(array, N);
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
 * Function:  shell_sort 
 * --------------------
 * Sorts an integer array, using the shell sort algorithm
 *
 *  a: the array that will be sorted
 *  n: number of elements in the array
 *
 */

void shell_sort(int array[], int n) {
    int h, i, j, t;
    for (h = n; h /= 2;) {
        for (i = h; i < n; ++i) {
            t = array[i];
            for (j = i; j >= h && t < array[j - h]; j -= h) {
                array[j] = array[j - h];
            }
            array[j] = t;
        }
    }
}