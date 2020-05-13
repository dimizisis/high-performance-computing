#include <stdio.h>
#include <math.h>
#include <omp.h>
 
#define EPSILON 1.0e-15

double calculate_epsilon(void);
 
int main(void) {

    double e;
    double start, end;
    
    start = omp_get_wtime();

    e = calculate_epsilon();
    
    end = omp_get_wtime();

    (void) printf("\ne = %.15f\n\n", e);

    (void) printf("Time elapsed: %g seconds\n", (end-start));

    return 0;
}

double calculate_epsilon(void){
    unsigned long long fact=1;
    double e=2.0, e0;
    int n=2, iterations=1;

    #pragma omp parallel default(shared) private(e0) reduction(+:e)
    {
        do {
            ++iterations;
            e0 = e;
            fact *= n++;
            e += 1.0 / fact;
        } while (fabs(e-e0) >= EPSILON);
    }

    return e;
}