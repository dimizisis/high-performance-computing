
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

__global__ void best_shuffle(const int *s, int *r, int *diff, int n){
    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < n){
        int i, j = 0, max = 0, cnt[128] = {0};
        char buf[256] = {0};

        for(i = 0; i < n; ++i)
            if (++cnt[(int)s[i]] > max) max = cnt[(int)s[i]];
        
        for(i = 0; i < 128; ++i)
            while (cnt[i]--) buf[j++] = i;
        
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
