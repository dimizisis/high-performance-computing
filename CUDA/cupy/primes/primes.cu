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
