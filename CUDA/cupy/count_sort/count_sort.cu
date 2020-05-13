__global__ void count_sort(int *a, int *s_a, int n) {
	
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;
        
    int i, j, count;
    for (i = index; i < n; i+=total_threads) {
        count = 0;
        for (j = 0; j < n; ++j)
            if (a[j] < a[i])
                ++count;
            else if (a[j] == a[i] && j < i)
                ++count;
        s_a[count] = a[i];
    }
}