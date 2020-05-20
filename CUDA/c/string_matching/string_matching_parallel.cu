#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 512
 
/* returns 0 if no match, 1 if matched, -1 if matched and at end */
__device__ int s_cmp(const char *s1, const char *s2){
    char c1 = 0, c2 = 0;
    while (c1 == c2) {
            c1 = *(s1++);
            if ('\0' == (c2 = *(s2++)))
                    return c1 == '\0' ? -1 : 1;
    }
    return 0;
}

__global__ void s_match(const char *s1, const char *s2){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // int total_threads = gridDim.x * blockDim.x;
 
    if (s1[idx] != '\0') {
            switch (s_cmp(s1 + idx, s2)) {
                case -1:
                {
                        printf("matched: pos %d (at end)\n", idx);
                        return;
                }
                case 1:
                {
                        printf("matched: pos %d\n", idx);
                        break;
                }
            }
    }
}
 
int main(int argc, char *argv[]){

    if (argc != 3){
        printf("Usage: %s <string 1> <string 2>", argv[0]);
        exit(-1);
    }

    size_t s1_len = strlen(argv[1]), s2_len = strlen(argv[2]);  /* length of input  */
    size_t s1_sz_mem = s1_len + 1, s2_sz_mem = s2_len + 1;        /* memory required  */

    /*
    * Host's arrays
    */
    char *s1, *s2;
    s1 = (char*) malloc(s1_sz_mem * sizeof(char));
    s2 = (char*) malloc(s2_sz_mem * sizeof(char));

    if (!s1 || !s2) {    /* validate memory created successfully or throw error */
        fputs ("error: name allocation failed, exiting.", stderr);
        return 1;
    }

    /*
    * Copy from arguments
    */
    strcpy(s1, argv[1]);
    strcpy(s2, argv[2]);

    printf("matching %s with %s:\n", s1, s2);

    float total_time, comp_time;
    cudaEvent_t total_start, total_stop, comp_start, comp_stop;
    cudaEventCreate(&total_start);
  	cudaEventCreate(&total_stop);
  	cudaEventCreate(&comp_start);
  	cudaEventCreate(&comp_stop);

    /*
    * Start counting total time
    */
    cudaEventRecord(total_start);

    /*
    * Device's array
    */
    char *dev_s1, *dev_s2;
    cudaMalloc(&dev_s1, s1_sz_mem*sizeof(char));
    cudaMalloc(&dev_s2, s2_sz_mem*sizeof(char));

    /*
	 * Copy c from host memory to host device memory
	 */
    cudaMemcpy(dev_s1, s1, s1_sz_mem*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_s2, s2, s2_sz_mem*sizeof(char), cudaMemcpyHostToDevice);

    /*
    * Start counting compile time
    */
    cudaEventRecord(comp_start);

    /*
    * Create blocks
    */
    int blocks = (s1_len+s2_len + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

    /*
    * Kernel call
    */
    s_match<<<blocks, THREADS_PER_BLOCK>>>(dev_s1, dev_s2);

    /*
    * Compile time count
    */
    cudaEventRecord(comp_stop);
    cudaEventSynchronize(comp_stop);
    cudaEventElapsedTime(&comp_time, comp_start, comp_stop);
    
    /*
    * Total time count
    */        
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_time, total_start, total_stop);

	/*
	 * Free memory on device
	 */
    cudaFree(dev_s1);
    cudaFree(dev_s2);
    cudaEventDestroy(comp_start);
    cudaEventDestroy(comp_stop);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);

    /*
    * GPU timing
    */
    printf("blocks: %d, total_threads: %d\n", blocks, THREADS_PER_BLOCK*blocks);
    printf("Total time (ms): %f\n", total_time);
    printf("Kernel time (ms): %f\n", comp_time);
    printf("Data transfer time (ms): %f\n", total_time-comp_time);
    
    return 0;
}