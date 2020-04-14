#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define N 500000
#define ROOT 0
#define TAG 100 /* for MPI_Send */

void rand_init_array(int array[], int n, int upper, int lower);
void display_array(int array[], int n);
void calculate_displacements(int* displacements, int* sendcounts, int size);
void calculate_sendcounts(int* sendcounts, int size, int n);
void shell_sort(int a[], int n);
int* merge(int *x, int n1, int *y, int n2);
int calculate_count(int rank, int* sendcounts, int received);

int main (int argc, char *argv[]) {
    
    int array[N];
    int rank, size;
    MPI_Status status;
    double begin, end;

    MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *sendcounts = malloc(sizeof(int)*size);
    int *displs = malloc(sizeof(int)*size);

    /*  Calculate send counts and displacements (for MPI_Scatterv)  */
    calculate_sendcounts(sendcounts, size, N);
    calculate_displacements(displs, sendcounts, size);

    if (rank == ROOT){
        rand_init_array(array, N, 0, N*4);
        // (void) printf("Initial array: ");
        // display_array(array, N);
        begin = MPI_Wtime();
    }
    
    /* Each thread will have different local_locations vector */
    int* local_array = (int*)malloc(sendcounts[rank]*sizeof(int));
    if (local_array == NULL) {printf ("Memory error\n"); MPI_Finalize(); return 4;}

    MPI_Scatterv(array, sendcounts, displs, MPI_INT, local_array, sendcounts[rank], MPI_INT, ROOT, MPI_COMM_WORLD);

    /* Perform Shell Sort only on the local array with size chunk (every process) */
    shell_sort(local_array, sendcounts[rank]);  

    if (rank == ROOT)
        /* Root process will only send to the next its array */
        MPI_Send(local_array, sendcounts[rank], MPI_INT, rank+1, TAG, MPI_COMM_WORLD);  

    else {  /* if process is not the root */
  
        int received_count = calculate_count(rank, sendcounts, 1);
        int* recv_array = (int*)malloc(received_count*sizeof(int));
        if (recv_array == NULL) {printf ("Memory error\n"); MPI_Finalize(); return 4;}
        
        /* Receive array from previous process */
        MPI_Recv(recv_array, received_count, MPI_INT, rank-1, 100, MPI_COMM_WORLD, &status);

        /* merged_array is the array the process is going to send (size of merged_array: send_count) */
        int send_count = calculate_count(rank, sendcounts, 0);

        /* Merge your array with the received one (merged_array)*/
        int* merged_array = merge(local_array, sendcounts[rank], recv_array, received_count);

        /* Sort merged array */
        shell_sort(merged_array, send_count);

        /* Send the merged array to the next process (when last is last process turn, will send array to root because of modulo) */
        MPI_Send(merged_array, send_count, MPI_INT, (rank+1)%size, TAG, MPI_COMM_WORLD);

    }

    /* Receive & print results */
    if(rank == ROOT) {
        int* results = (int*)malloc(sizeof(int)*N);
        if (results == NULL) {printf ("Memory error\n"); MPI_Finalize(); return 4;}

        MPI_Recv(results, N, MPI_INT, size-1, TAG, MPI_COMM_WORLD, &status);
        end = MPI_Wtime();
        // (void) printf("Sorted Array: ");
        // display_array(results, N);
        (void) printf("Time elapsed: %g\n", (end-begin));
    }

    MPI_Finalize();
    
    return 0;
}

/*
 * Function:  calculate_displacements 
 * --------------------
 * Calculates displacements (for MPI_Scatterv) and puts them on an array
 *
 *  displacements: pointer of the array that will be filled with potential displacements
 *  sendcounts: pointer of the array which contains the size of all local arrays of each process (size of the array is the number of processes)
 *  size: number of processes
 *
 */

void calculate_displacements(int* displacements, int* sendcounts, int size){
    int i, sum=0;
    for (i=0; i < size; ++i) {
        displacements[i] = sum;
        sum += sendcounts[i];
    }
}

/*
 * Function:  calculate_sendcounts 
 * --------------------
 * Calculates sendcounts (how many elements we're going to send to each process, for MPI_Scatterv) and puts them on an array
 *
 *  sendcounts: pointer of the array which contains the size of all local arrays of each process (size of the array is the number of processes)
 *  size: number of processes
 *  n: number of elements of initial array
 *
 */

void calculate_sendcounts(int* sendcounts, int size, int n){
    int i, extra = N % size;
    for (i=0; i < size; ++i) {
        sendcounts[i] = N/size;
        if (extra > 0) {
            ++sendcounts[i];
            --extra;
        }
    }
}

/*
 * Function:  calculate_count 
 * --------------------
 * Calculates send or receive count (how many elements we're going to receive from previous process or send to the next one)
 *
 *  rank: the rank of the process
 *  sendcounts: pointer of the array which contains the size of all local arrays of each process (size of the array is the number of processes)
 *  received: 1 if we are calculating received count, 0 if we are calculating send count
 * 
 * Returns: the asked count (int)
 *
 */

int calculate_count(int rank, int* sendcounts, int received){
    int i, count=0;
    for(i=0;i<=rank-received;++i)
        count += sendcounts[i];

    return count;
}

/*
 * Function:  merge 
 * --------------------
 * Merges two arrays with different length
 *
 *  x: pointer of the first array
 *  n1: size of array x
 *  y: pointer of the second array
 *  n2: size of array y
 * 
 * Returns: pointer of the merged array
 *
 */

int* merge(int *x, int n1, int *y, int n2){
	int i=0, j=0, k=0;
	int* result = (int*)malloc((n1+n2)*sizeof(int));;

	while(i < n1 && j < n2)
		if (x[i] < y[j]){
			result[k] = x[i];
			++i; 
            ++k;
		}
		else{
			result[k] = y[j];
			++j; 
            ++k;
		}
	if (i == n1)
		while(j < n2){
			result[k] = y[j];
			++j;
            ++k;
		}
	else
		while(i < n1){
			result[k] = x[i];
			++i;
            ++k;
		}

	return result;
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

void shell_sort(int a[], int n){
    int h, i, j, t;
    for (h=n; h /= 2;) {
        for (i=h; i < n; ++i) {
            t = a[i];
            for (j=i; j >= h && t < a[j - h]; j -= h)
                a[j] = a[j - h];
            a[j] = t;
        }
    }
}
