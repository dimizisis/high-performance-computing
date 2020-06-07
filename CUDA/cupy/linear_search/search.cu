
__global__ void lsearch(int *a, int n, int x, int *index){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        if (a[i] == x)
            index[0] = i;
}
