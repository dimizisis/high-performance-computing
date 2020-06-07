#include <stdio.h>
#include <cuda.h>
#include <assert.h>

#define N 10000000
#define UPPER N*4
#define LOWER 1
#define THREADS_PER_BLOCK 512

void rand_init_array(int *array, int n, int upper, int lower);
void display_array(int *array, int n);

__global__ void lsearch(int *a, int n, int x, int *index){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        if (a[i] == x)
            index[0] = i;
}

int main(void) {

    /* Allocation & Initialization */
    int *array = (int*) malloc(N*sizeof(int));
    rand_init_array(array, N, UPPER, LOWER);
    // display_array(array, N);
    int *index = (int *)malloc(1*sizeof(int));
    index[0] = -1;
    int key = 2;
    /* --------------------------- */

    int blocks;	
    float total_time, comp_time;
    cudaEvent_t total_start, total_stop, comp_start, comp_stop;
    cudaEventCreate(&total_start);
  	cudaEventCreate(&total_stop);
  	cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_stop);

    /*
	 * Memory allocation on device
	 */
    int *array_dev, *index_dev;
    cudaMalloc((void **)&array_dev, N*sizeof(int));
    cudaMalloc((void **)&index_dev, 1*sizeof(int));

    cudaEventRecord(total_start);

    /*
	 * Copy array from host memory to device memory
	 */
    cudaMemcpy(array_dev, array, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(index_dev, index, 1*sizeof(int), cudaMemcpyHostToDevice);
    
    /*
     * Create sufficient blocks 
     */
    blocks = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

    cudaEventRecord(comp_start);

    /*
     * Kernel call
     */ 
    lsearch<<< blocks, THREADS_PER_BLOCK >>>(array_dev, N, key, index_dev);

    cudaEventRecord(comp_stop);
    cudaEventSynchronize(comp_stop);
    cudaEventElapsedTime(&comp_time, comp_start, comp_stop);

    /*
	 * Copy c from host device memory to host memory
	 */
    cudaMemcpy(index, index_dev, 1*sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_time, total_start, total_stop);

    /*
	 * Free memory on device
	 */
	cudaFree(array_dev);
    cudaFree(index_dev);

    /*
     * GPU timing
     */
    printf("N: %d, blocks: %d, total_threads: %d\n", N, blocks, THREADS_PER_BLOCK*blocks);
    printf("Total time (ms): %f\n", total_time);
    printf("Kernel time (ms): %f\n", comp_time);
    printf("Data transfer time (ms): %f\n", total_time-comp_time);
    printf("\nNum %d is at index %d\n", key, index[0]);
    return 0;
}

int cmpfunc(const void * a, const void * b){
	// Compare function used by qsort
	return ( *(int*)a - *(int*)b );
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

void rand_init_array(int *array, int n, int upper, int lower){
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

void display_array(int *array, int n){
    (void) printf("[ ");
    int i;
    for (i=0; i<n; ++i)
        (void) printf("%d ", array[i]);
    (void) printf("]\n\n");
}