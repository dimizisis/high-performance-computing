#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 500000

int max(int array[], int n, int i, int j, int k);
void downheap(int array[], int n, int i);
void heapsort(int array[], int n);
void rand_init_array(int array[], int n, int upper, int lower);
void display_array(int array[], int n);
void swap(int array[], int i, int j);

int main(void) {
    int a[N];
    rand_init_array(a, N, 1, 10);

    // /* Printing */
    // (void) printf("Initial Array: ");
    // display_array(a, N);

    clock_t start = clock();

    heapsort(a, N); // sorting...

    clock_t end = clock();

    (void) printf("\nTime elapsed for sorting: %g seconds.\n", (double)(end-start) / CLOCKS_PER_SEC);

    // /* Printing */
    // (void) printf("Sorted Array: ");
    // display_array(a, N);

    return 0;
}

/*
 * Function:  rand_init_array 
 * --------------------
 * Fills an integer array with random numbers
 *
 *  array: the integer array that will be filled with numbers
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
 *  array: the integer array that will be printed
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
 * Function:  max 
 * --------------------
 * Finds max between 3 elements of an array
 *
 *  array: the integer array that will be sorted
 *  n: number of elements in the array
 *  i: index i (from downheap)
 *  j: index j == 2*i+1 (from downheap)
 *  k: index k == 2*i+2 (from downheap)
 *
 */

int max(int array[], int n, int i, int j, int k){
    int m = i;
    if (j < n && array[j] > array[m])
        m = j;
    if (k < n && array[k] > array[m])
        m = k;
    return m;
}

/*
 * Function:  downheap 
 * --------------------
 * Performs downheap (while sorting)
 *
 *  array: the integer array that will be sorted
 *  n: number of elements in the array
 *  i: index i (from heapsort)
 *
 */
 
void downheap(int array[], int n, int i){
    while (1) {
        int j = max(array, n, i, 2*i+1, 2*i+2);
        if (j == i)
            break;
        swap(array, i, j);
        i = j;
    }
}

/*
 * Function:  swap 
 * --------------------
 * Swaps two elements of an array
 *
 *  array: the integer array that will be printed
 *  n: number of elements in the array
 *  i: index in which element array[i] is located (and array[j] will be located after swap)
 *  j: index in which element array[j] is located (and array[i] will be located after swap)
 *
 */

void swap(int array[], int i, int j){
    int tmp = array[i];
    array[i] = array[j];
    array[j] = tmp;
}

/*
 * Function:  heapsort 
 * --------------------
 * Sorts an integer array, using the heapsort algorithm
 *
 *  array: the array that will be sorted
 *  n: number of elements in the array
 *
 */
 
void heapsort(int array[], int n){
    int i;
    for(i=(n-2)/2; i>=0;--i) 
        downheap(array, n, i);

    for(i=0; i<n; ++i) {
        swap(array, n-i-1, 0);
        downheap(array, n-i-1, 0);
    }
}