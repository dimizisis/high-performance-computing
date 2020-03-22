#include <stdio.h>
#include <math.h>
#include <omp.h>

#define NSTEPS 134217728

double calc_pi(double *pi);

int main(int argc, char** argv){
    
    double pi;
    double start, end;
    double ref_pi = 4.0 * atan(1.0);

    start = omp_get_wtime();

    calc_pi(&pi);

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
 *  pi: pointer to variable in which pi will be stored
 *
 */

double calc_pi(double *pi){
    long i;
    double dx = 1.0 / NSTEPS;

    #pragma omp parallel default(shared) private(i)
    {
        #pragma omp for
        for (i=0; i<NSTEPS; ++i){
            double x = (i + 0.5) * dx;
            #pragma omp atomic
            *pi += 1.0 / (1.0 + x * x);
        }

    }
    
    *pi *= 4.0*dx;

}