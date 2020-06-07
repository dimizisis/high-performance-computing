#include <stdio.h>
#include <stdlib.h>

#define N 6
#define UPPER 1
#define LOWER N*3

int is_sorted(int *a, int n);
void shuffle(int *a, int n);
void bogo_sort(int *a, int n);
void swap_random(int *a, int i, int n);
void rand_init_array(int *a, int n, int upper, int lower);
void display_array(int *a, int n);

int main(void){
    int a[N];
    rand_init_array(a, N, UPPER, LOWER);
    bogo_sort(a, N);
    display_array(a, N);
    return 0;
}

/*
 * Function:  rand_init_array 
 * --------------------
 * Fills an integer array with random numbers
 *
 *  a: the array that will be filled with numbers
 *  n: number of elements in the array
 *  upper: highest value of random number
 *  lower: lowest value of random number
 *
 */

void rand_init_array(int *a, int n, int upper, int lower){
    int i;    
    for (i=0; i<n; ++i)
        a[i] = (rand() % (upper - lower + 1)) + lower;
}

/*
 * Function:  is_sorted 
 * --------------------
 * Checks if array is sorted
 *
 *  a: the array (integer)
 *  n: number of elements in the array
 *
 */

int is_sorted(int *a, int n){
    while ( --n >= 1 )
        if ( a[n] < a[n-1] ) return 0;
    return 1;
}

/*
 * Function:  shuffle 
 * --------------------
 * Randomizes elements of array
 *
 *  a: the array (integer)
 *  n: number of elements in the array
 *
 */

void shuffle(int *a, int n){
    int i, t, r;
    for(i=0; i < n; ++i)
        swap_random(a, i, n);
}

/*
 * Function:  swap_random 
 * --------------------
 * Randomizes elements of array
 *
 *  a: the array (integer)
 *  i: the index of element that will be swapped
 *  n: number of elements in the array
 *
 */

void swap_random(int *a, int i, int n){
    int t, r;
    t = a[i];
    r = rand() % n;
    a[i] = a[r];
    a[r] = t;
}

/*
 * Function:  bogo_sort 
 * --------------------
 * Performs bogo sort (random suffle until the array is sorted)
 *
 *  a: the array (integer)
 *  n: number of elements in the array
 *
 */

void bogo_sort(int *a, int n){
    while ( !is_sorted(a, n) ) shuffle(a, n);
}

/*
 * Function:  display_array 
 * --------------------
 * Prints an integer array to user
 *
 *  a: the array that will be printed
 *  n: number of elements in the array
 *
 */

void display_array(int *a, int n){
    int i;
    for (i=0; i < N; ++i) printf("%d ", a[i]);
    printf("\n");
}