#include <stdio.h>
#include <math.h>
#include <omp.h>

#define NSTEPS 134217728

double calc_pi(void);

int main(int argc, char** argv){
    
    double pi;
    double start, end;
    double ref_pi = 4.0 * atan(1.0);

    start = omp_get_wtime();

    pi = calc_pi();

    end = omp_get_wtime();

    (void) printf("\npi with %ld steps is %.10f in %.6f seconds (error = %e)\n",
           NSTEPS, pi, (end-start), fabs(ref_pi - pi));

    return 0;
}

/*
 * Function:  calc_pi 
 * --------------------
 * Calculates pi
 *
 */

double calc_pi(void){
    long i;
    double dx = 1.0 / NSTEPS;
    double pi = 0.0;

    for (i=0; i<NSTEPS; ++i){
        double x = (i + 0.5) * dx;
        pi += 1.0 / (1.0 + x * x);
    }

    pi *= 4.0 * dx;

    return pi;
}