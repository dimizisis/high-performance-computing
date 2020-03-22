#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 90000
#define UPPER N*4
#define LOWER 1

void rand_init_array(int array[], int n, int upper, int lower);
void display_array(int array[], int n);
void count_sort(int a[], int n);
void display_time(double start, double end);

int main(void){ 

    int array[N];

    rand_init_array(array, N, UPPER, LOWER);

    // (void) printf("Initial array: ");
    // display_array(array, N);

    (void) printf("Sorting began...\n\n");
    double begin = omp_get_wtime();
    count_sort(array, N);
    double end = omp_get_wtime();

    display_time(begin, end);
    
    // (void) printf("\n\nSorted array: ");
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
 * Function:  count_sort 
 * --------------------
 * Sorts an integer array, using the count sort algorithm (enumeration sort)
 *
 *  a: the array that will be sorted
 *  n: number of elements in the array
 *
 */

void count_sort(int a[], int n) {
    int i, j, count;
    int* temp = malloc(n*sizeof(int));
    #pragma omp parallel for private(count,i,j) shared(a)
    for (i = 0; i < n; ++i) {
        count = 0;
        for (j = 0; j < n; ++j)
            if (a[j] < a[i])
                count++;
            else if (a[j] == a[i] && j < i)
                count++;
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
    (void) printf("Time spent for sorting: %f seconds\n", (end-start));
}
