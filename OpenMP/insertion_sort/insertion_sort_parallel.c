#include <stdio.h>
#include <omp.h>
#include <stdlib.h> 

#define N 100000

void insertion_sort (int a[], int n);
void rand_init_array(int array[], int n, int upper, int lower);
void display_array(int array[], int n);
void display_time(double start, double end);

int main(void) {

    int a[N];
    double start_time, end_time;
    rand_init_array(a, N, 1, 12);  // generate random integers in array 
    (void) printf("Initial Array: ");
    // display_array(a, N);    // display initial array
    start_time = omp_get_wtime();
    insertion_sort(a, N);   // sorting...
    end_time = omp_get_wtime();
    (void) printf("Sorted Array: ");
    // display_array(a, N);    // display sorted array

    display_time(start_time, end_time); // display time elapsed for sorting

    return 0;
}

/*
 * Function:  insertion_sort 
 * --------------------
 * Sorts an integer array, using the insertion sort algorithm
 *
 *  a: the array that will be sorted
 *  n: number of elements in the array
 *
 */

void insertion_sort(int a[], int n) {
    int i, j, t;
    #pragma omp parallel default(shared) private(i, j, t)
    {
        for (i=1; i<n; ++i) {
            t = a[i];
            #pragma omp critical
            {
                for (j = i; j > 0 && t < a[j - 1]; --j)
                    a[j] = a[j - 1];
            }
            a[j] = t;
        }
    }
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
 * Function:  display_time 
 * --------------------
 * Prints the time (seconds) elapsed for sorting
 *
 *  start: the time (seconds) in which the sorting started
 *  end: the time (seconds) in which the sorting finished
 *
 */

void display_time(double start, double end){
    (void) printf("Time spent for sorting: %f seconds\n", (end-start));
}