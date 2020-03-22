#include <stdio.h>
#include <math.h>
#include <omp.h>

#define NSTEPS 134217728

double calc_pi(double pi[], int n);
void zeros(double array[], int n);

int main(int argc, char** argv){
    
    double start, end;
    double ref_pi = 4.0 * atan(1.0);
    int num_threads = omp_get_num_threads();
    double pi, pi_array[num_threads];

    zeros(pi_array, num_threads);

    start = omp_get_wtime();

    pi = calc_pi(pi_array, num_threads);

    end = omp_get_wtime();

    (void) printf("\npi with %ld steps is %.10f in %g seconds (error = %e)\n",
           NSTEPS, pi, (end-start), fabs(ref_pi - pi));

    return 0;
}

/*
 * Function:  calc_pi 
 * --------------------
 * Calculates pi
 *
 *  pi: the array that will be filled with each thread's calculations
 *  n: number of elements in the array (num of threads)
 *
 */

double calc_pi(double pi[], int n){
    long i;
    int j;
    double dx = 1.0 / NSTEPS;
    double calculated_pi = 0.0;
    int tid;

    #pragma omp parallel default(shared) num_threads(n) private(i, j, tid)
    {
        tid = omp_get_thread_num();
        #pragma omp for
        for (i=0; i<NSTEPS/n; ++i){
            double x = (i + 0.5) * dx;
            pi[tid] += 1.0 / (1.0 + x * x);
        }

        #pragma omp barrier
        for (j=0; j<n;++j)
            calculated_pi += pi[j];

    }

    calculated_pi *= 4.0*dx;
    
    return calculated_pi;

}

/*
 * Function:  zeros 
 * --------------------
 * Initializes an integer array with zeros
 *
 *  array: the array that will be filled with zeros
 *  n: number of elements in the array
 *
 */

void zeros(double array[], int n){
	int j;
	for (j=0; j<n; ++j)
		array[j]=0.0;
}