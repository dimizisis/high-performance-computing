#include <stdio.h>

int lsearch(int *a, int n, int x);

int main(void) {
    int a[] = {-31, 0, 1, 2, 2, 4, 65, 83, 99, 782};
    int n = sizeof a / sizeof a[0];
    int x = 2;
    int i = lsearch(a, n, x);
    printf("%d is at index %d\n", x, i);
    return 0;
}

int lsearch(int *a, int n, int x){
    int i, index=-1;
    for(i=0;i<n;++i){
        if (a[i] == x){
            index = i;
            break;
        }
    }
    return index;
}
