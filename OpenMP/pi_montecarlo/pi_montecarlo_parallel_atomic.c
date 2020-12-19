/* Program to compute Pi using Monte Carlo methods */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#define SEED 35791246

int main(int argc, char* argv[])
{
   int niter=0;
   double x,y;
   int i,count=0; /* # of points in the 1st quadrant of unit circle */
   double z;
   double pi;

   if (argc != 2) {
		printf ("Usage : %s <num points>\n", argv[0]);
                exit(1);
	}
   niter = strtol(argv[1], NULL, 10);

   double start = omp_get_wtime();

   /* initialize random numbers */
   srand(SEED);
   #pragma omp parallel for default(shared), private(i, x, y, z)
   for (i=0; i<niter; ++i) {
      x = (double)rand()/RAND_MAX;
      y = (double)rand()/RAND_MAX;
      z = x*x+y*y;
      if (z<=1){
         #pragma omp atomic
         ++count;
      } 
   }

   pi=(double)count/niter*4;

   double stop = omp_get_wtime();

   (void) printf("Completed in: %g seconds\n", (stop-start));

   (void) printf("# of trials= %d , estimate of pi is %g \n",niter,pi);

   return 0;
}
