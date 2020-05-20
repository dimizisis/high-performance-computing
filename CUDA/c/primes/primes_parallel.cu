#include <stdio.h>
#include <stdlib.h>
#define N 20
#define THREADS_PER_BLOCK 512
#define BLOCKS (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK

__global__ void find_primes(int *a, int n) { 

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // int total_threads = gridDim.x * blockDim.x;
    int is_prime = 1;

    if (idx > 1 && idx < n){
        int j;
        for (j=2; j<idx/2+1; ++j){
            if (!(idx % j) && j != idx){
                is_prime = 0;
                break;
            }
        }
        if (is_prime) a[idx] = 1;
        is_prime = 1;
    }

}

int main(int argc, char *argv[]) {
  
    float total_time, comp_time;
    cudaEvent_t total_start, total_stop, comp_start, comp_stop;
    cudaEventCreate(&total_start);
  	cudaEventCreate(&total_stop);
  	cudaEventCreate(&comp_start);
  	cudaEventCreate(&comp_stop);

    /*
    * Host's array
    */
    int *array;
    array = (int*) calloc(N, sizeof(int));

    /*
    * Start counting total time
    */
    cudaEventRecord(total_start);

    /*
    * Device's array
    */
    int *dev_array;
    cudaMalloc(&dev_array, N * sizeof(int));

    /*
    * Start counting compile time
    */
    cudaEventRecord(comp_start);

    /*
    * Kernel call
    */
    find_primes<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_array, N);

    /*
    * Compile time count
    */

    cudaEventRecord(comp_stop);
    cudaEventSynchronize(comp_stop);
    cudaEventElapsedTime(&comp_time, comp_start, comp_stop);

    /*
	 * Copy c from host device memory to host memory
	 */
    cudaMemcpy(array, dev_array, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    /*
    * Total time count
    */        
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_time, total_start, total_stop);

	/*
	 * Free memory on device
	 */
    cudaFree(dev_array);
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

    /*
    * Printing primes
    */
    (void) printf("\n\nPrimes: [ ");
    int i;
    for (i=2; i<=N; ++i)
        if (array[i])
            (void) printf("%d ", i);
    (void) printf("]\n\n");

    return 0;
}