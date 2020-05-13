__global__ void count_characters(int *buffer, int *freq, long file_size, int base) {
	
    int index = threadIdx.x + blockIdx.x * blockDim.x;     
    int total_threads = gridDim.x * blockDim.x;
    
    long i;
    for (i=index; i<file_size; i+=total_threads)
        atomicAdd(&(freq[buffer[i] - base]), 1);
        
}