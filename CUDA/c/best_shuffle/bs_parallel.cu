#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 256

__global__ void best_shuffle(const char *s, char *r, int *diff, int n);
__device__ void update_buf(int *cnt, char *buf);
__device__ int find_max(const char *s, int *cnt, int n);

char * get_input_word(int argc, char *argv[]);

/*
 * Main
 */

int main(int argc, char *argv[]){
        
    char *t = get_input_word(argc, argv);
    printf("\nword: %s\n", t);
    int n = strlen(t);
    int blocks, *diff;
    char *r;

    float total_time, comp_time;
    cudaEvent_t total_start, total_stop, comp_start, comp_stop;
    cudaEventCreate(&total_start);
  	cudaEventCreate(&total_stop);
  	cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_stop);

    /*
     * Memory allocation on host
     */
    r = strdup(t);
    diff = (int *)calloc(1, sizeof(int));

    /*
     * Memory allocation on device
     */
    char *t_dev, *r_dev;
    int *diff_dev;
    cudaMalloc(&r_dev, strlen(t)*sizeof(char));
    cudaMalloc(&t_dev, strlen(t)*sizeof(char));
    cudaMalloc(&diff_dev, 1*sizeof(int));
      
    cudaEventRecord(total_start);
  
    /*
    * Copy array from host memory to device memory
    */
    cudaMemcpy(t_dev, t, strlen(t)*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(r_dev, t, strlen(t)*sizeof(char), cudaMemcpyHostToDevice);
 
    cudaEventRecord(comp_start);

    /*
    * Create sufficient blocks
    */
    blocks = (n + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
 
    /*
    * Kernel call
    */ 
    best_shuffle<<< blocks, THREADS_PER_BLOCK >>>(t_dev, r_dev, diff_dev, n);
  
    cudaEventRecord(comp_stop);
    cudaEventSynchronize(comp_stop);
    cudaEventElapsedTime(&comp_time, comp_start, comp_stop);
  
    /*
    * Copy c from host device memory to host memory
    */
    cudaMemcpy(r, r_dev, strlen(t)*sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(diff, diff_dev, 1*sizeof(int), cudaMemcpyDeviceToHost);
     
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_time, total_start, total_stop);
 
    /*
    * Free memory on device
    */
    cudaFree(diff_dev);
    cudaFree(r_dev);
    cudaFree(t_dev);
    cudaEventDestroy(comp_start);
    cudaEventDestroy(comp_stop);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
         
    /*
    * GPU timing
    */
    printf("N: %d, blocks: %d, total_threads: %d\n", n, blocks, THREADS_PER_BLOCK*blocks);
    printf("Total time (ms): %f\n", total_time);
    printf("Kernel time (ms): %f\n", comp_time);
    printf("Data transfer time (ms): %f\n", total_time-comp_time);

    /*
    * Initial word, final word & the difference between them
    */
    printf("\n%s %s (%d)\n", t, r, diff[0]);

    free(r);
    free(diff);
	return 0;
}

/*
 * Function:  find_max 
 * --------------------
 * Finds the letter with maximum frequency among the other characters of the string
 *
 *  s: pointer of the char array (constant)
 *  cnt: int array, which has the characters' counter role
 *  n: number of characters of the word
 *
 */

__device__ int find_max(const char *s, int *cnt, int n){
    int i, max=0;
    for(i = 0; i < n; ++i)
        if (++cnt[(int)s[i]] > max) max = cnt[(int)s[i]];
    return max;
}

/*
 * Function:  update_buf 
 * --------------------
 * Updates buf array (char), according to counter array (cnt). Buf will be used in deterministic function
 *
 *  cnt: int array, which has the characters' counter role
 *  buf: char array, the one we are updating
 *
 */

__device__ void update_buf(int *cnt, char *buf){
    int i, j=0;
    for(i = 0; i < 128; ++i)
        while (cnt[i]--) buf[j++] = i;
}

/*
 * Function:  best_shuffle 
 * --------------------
 * Shuffles given string (char array) with the use of a deterministic function
 *
 *  s: pointer of the char array (constant)
 *  r: pointer of a copy of char array (will contain final string)
 *  diff: pointer of int array, which will contain the difference between initial & final string
 *  n: number of characters of the word
 *
 */

__global__ void best_shuffle(const char *s, char *r, int *diff, int n){
    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < n){
        int i, max, cnt[128] = {0};
        char buf[256] = {0};
        
        max = find_max(s, cnt, n);
        update_buf(cnt, buf);
        
        for(i = 0; i < n; ++i){
            if (r[idx] == buf[i]) {
                r[idx] = buf[(i + max) % n] & ~128;
                buf[i] |= 128;
                break;
            }
        }

        atomicAdd(&(diff[0]), diff[0]+(r[idx] == s[idx]));

    }
}

/*
 * Function:  get_input_word 
 * --------------------
 * Returns the input word the user entered (to be shuffled). If no word inserted, throws an error.
 *
 *  argc: number of arguments
 *  argv: the actual arguments
 *
 */

char * get_input_word(int argc, char *argv[]){
    if (argc != 2){
        printf("Usage: %s \"<input_word>\"", argv[0]);
        exit(1);
    }
    return argv[1];
}