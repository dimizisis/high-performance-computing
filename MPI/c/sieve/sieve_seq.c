#include <stdio.h>
#include <malloc.h>
#include <time.h>

#define N 2000

void init_array(int* a, int n);
void sieve(int* a, int n);
void print_primes(int array[], int n);
void display_time(double start, double end);
 
int main(int argc, char *argv){
    int *array;
    clock_t begin, end;
    array =(int *)malloc((N + 1) * sizeof(int));
    init_array(array, N);
    begin = clock();
    sieve(array,N);
    end = clock();
    // (void) printf("\nPrimes numbers from 1 to %d are : ", N+1);
    // print_primes(array, N+1);
    display_time((double)begin, (double)end);
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
    for(i=2; i<=n; ++i)
        a[i] = 1;
}

/*
 * Function:  sieve 
 * --------------------
 * Performs sieve
 *
 *  a: pointer of the array that contains ones
 *  n: number of elements in the array
 *
 */
 
void sieve(int *a, int n){
    int i, j;
    for(i=2; i<=n; i++)
        if(a[i] == 1)
            for(j=i; (i*j)<=n; j++)
                a[(i*j)] = 0;
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
    (void) printf("Time spent for sorting: %f seconds\n\n", (double)(end-start) / 1000000.0F * 1000) ;
}
