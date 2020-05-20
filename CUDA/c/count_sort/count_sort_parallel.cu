#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>

#define N 10
#define UPPER N*4
#define LOWER 1
#define THREADS_PER_BLOCK 512

/*
 * Kernel function
 */
__global__ void count_sort(int *a, int *s_a, int n) {
	
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;
    int count = 0;
        
    int j;
    for (j = 0; j < n; ++j)
        if (a[j] < a[index])
            ++count;
        else if (a[j] == a[index] && j < index)
            ++count;
    s_a[count] = a[index];

}

void rand_init_array(int *array, int n, int upper, int lower);
void display_array(int *array, int n);

/*
 * Main
 */

int main(int argc, char *argv[]){
    
    int blocks;	
    float total_time, comp_time;
    cudaEvent_t total_start, total_stop, comp_start, comp_stop;
    cudaEventCreate(&total_start);
  	cudaEventCreate(&total_stop);
  	cudaEventCreate(&comp_start);
  	cudaEventCreate(&comp_stop);

	/*
	 * Memory allocation on host 
	 */
    int *array = (int *)malloc(N*sizeof(int));
    int *sorted_array = (int *)malloc(N*sizeof(int));

    /*
	 * Init array
	 */
    rand_init_array(array, N, UPPER, LOWER);
    display_array(array, N);

	/*
	 * Memory allocation on device
	 */
	int *array_dev, *sorted_dev;
    cudaMalloc((void **)&array_dev, N*sizeof(int));
    cudaMalloc((void **)&sorted_dev, N*sizeof(int));
	
    cudaEventRecord(total_start);

    /*
	 * Copy array from host memory to device memory
	 */
	cudaMemcpy(array_dev, array, N*sizeof(int), cudaMemcpyHostToDevice);
	
    /*
    * Create sufficient blocks 
    */
    blocks = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

    cudaEventRecord(comp_start);
	/*
    * Kernel call
    */ 
	count_sort<<< blocks, THREADS_PER_BLOCK >>>(array_dev, sorted_dev, N);

    cudaEventRecord(comp_stop);
    cudaEventSynchronize(comp_stop);
    cudaEventElapsedTime(&comp_time, comp_start, comp_stop);

	/*
	 * Copy c from host device memory to host memory
	 */
	cudaMemcpy(sorted_array, sorted_dev, N*sizeof(int), cudaMemcpyDeviceToHost);
	
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_time, total_start, total_stop);
	/*
	 * Free memory on device
	 */
	cudaFree(array_dev);
    cudaFree(sorted_dev);
    cudaEventDestroy(comp_start);
    cudaEventDestroy(comp_stop);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
       
    /*
    * GPU timing
    */
    printf("N: %d, blocks: %d, total_threads: %d\n", N, blocks, THREADS_PER_BLOCK*blocks);
    printf("Total time (ms): %f\n", total_time);
    printf("Kernel time (ms): %f\n", comp_time);
    printf("Data transfer time (ms): %f\n", total_time-comp_time);
    display_array(sorted_array, N);
        
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

