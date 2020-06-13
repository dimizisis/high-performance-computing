#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#define N 6
#define UPPER 1
#define LOWER N*4
#define THREADS_PER_BLOCK 1
#define BLOCKS 1 //(N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK

void rand_init_array(int *a, int n, int upper, int lower);
void display_array(int *a, int n);

__global__ void setup_kernel(curandState *state){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(clock64(), idx, 0, &state[idx]);
}

__device__ float getnextrand(curandState *state){

    return (float)(curand_uniform(state));
}
  
__device__ int getnextrandscaled(curandState *state, int scale){
  
    return (int) scale * getnextrand(state);
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

__device__ void swap_random(int *a, curandState *state){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int r = getnextrandscaled(state, N);
    a[r] = atomicExch(&(a[idx]), a[r]);
    printf("%d ", r);

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

__device__ int is_sorted(int *a, int n){
    while ( --n >= 1 )
        if ( a[n] < a[n-1] || a[n] == a[n-1]) return 0;
    return 1;
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

__global__ void bogo_sort(int *a, int n, int *found, curandState *state){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n){
        while(!found[0]){
            swap_random(a, state);
            found[0] = is_sorted(a, n);
        }
    }
}

/*
 * Main
 */

int main(int argc, char *argv[]){
    
    float total_time, comp_time;
    cudaEvent_t total_start, total_stop, comp_start, comp_stop;
    cudaEventCreate(&total_start);
  	cudaEventCreate(&total_stop);
  	cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_stop);

    /* Some initializations & allocations to generate random number within kernel */

    curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState));
    
    setup_kernel<<< BLOCKS, THREADS_PER_BLOCK >>>(d_state);
    /* -------------------------------------------------------------------------- */

    /*
	 * Memory allocation on host 
	 */
    int *array = (int *)malloc(N*sizeof(int));
    int *found = {0};

    /*
     * Init array
     */
    rand_init_array(array, N, UPPER, LOWER);
    display_array(array, N);
 
    /*
     * Memory allocation on device
     */
    int *array_dev, *found_dev;
    cudaMalloc((void **)&array_dev, N*sizeof(int));
    cudaMalloc((void **)&found_dev, 1*sizeof(int));
     
    cudaEventRecord(total_start);
 
    /*
     * Copy array from host memory to device memory
     */
    cudaMemcpy(array_dev, array, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(found_dev, found, 1*sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(comp_start);

    /*
     * Kernel call
     */ 
    bogo_sort<<< BLOCKS, THREADS_PER_BLOCK >>>(array_dev, N, found_dev, d_state);
 
    cudaEventRecord(comp_stop);
    cudaEventSynchronize(comp_stop);
    cudaEventElapsedTime(&comp_time, comp_start, comp_stop);
 
    /*
     * Copy c from host device memory to host memory
     */
    cudaMemcpy(array, array_dev, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_time, total_start, total_stop);

    /*
     * Free memory on device
     */
    cudaFree(array_dev);
    cudaEventDestroy(comp_start);
    cudaEventDestroy(comp_stop);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
        
    /*
     * GPU timing
     */
    printf("N: %d, blocks: %d, total_threads: %d\n", N, BLOCKS, THREADS_PER_BLOCK*BLOCKS);
    printf("Total time (ms): %f\n", total_time);
    printf("Kernel time (ms): %f\n", comp_time);
    printf("Data transfer time (ms): %f\n", total_time-comp_time);
    display_array(array, N);
         
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
    for (i=0; i < n; ++i) printf("%d ", a[i]);
    printf("\n");
}