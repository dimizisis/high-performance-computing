#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void main ( int argc, char *argv[] )  {

int   i, j, N, *flag;
float *x, *b, **a, sum;
char any;

	if (argc != 2) {
		printf ("Usage : %s <matrix size>\n", argv[0]);
                exit(1);
	}
	N = strtol(argv[1], NULL, 10);

	/* Allocate space for matrices */
	a = (float **) malloc ( N * sizeof ( float *) );
	for ( i = 0; i < N; i++) 
		a[i] = ( float * ) malloc ( N * sizeof ( float ) );
	b = ( float * ) malloc ( N * sizeof ( float ) );
	x = ( float * ) malloc ( N * sizeof ( float ) );
	flag = ( int *) malloc ( N * sizeof ( int ) );

	/* Create floats between 0 and 1. Diagonal elents between 2 and 3. */
	srand ( time ( NULL));
	for (i = 0; i < N; i++) {
		x[i] = 0.0;
		b[i] = (float)rand()/(RAND_MAX*2.0-1.0);
		flag[i] = 0 ;
		a[i][i] = 2.0+(float)rand()/(RAND_MAX*2.0-1.0);
		for (j = 0; j < i; j++) 
			a[i][j] = (float)rand()/(RAND_MAX*2.0-1.0);;
	} 

    /* Calulation */
    #pragma omp parallel for private (i, j, sum) shared (x, b, a, flag)
	for (i = 0; i < N; i++) {
		sum = 0.0;
		for (j = 0; j < i; j++) {
		    // while x[j] is not ready wait here check its flag 
		    // no need for atomic read 
		    #pragma omp flush(flag)
		    while (!flag[j]) { 
		    	#pragma omp flush (flag) 
		    }
			sum = sum + (x[j] * a[i][j]);
			//printf ("%d %d %f %f %f \t \n", i, j, x[j], a[i][j], sum);
		}	
		// only one thread writes to x[i]
		x[i] = (b[i] - sum) / a[i][i];
		// x[i] is ready so change its flag
		#pragma omp atomic write
		flag[i] = 1;
		#pragma omp flush (flag)
		//printf ("%d %f %f %f %f \n", i, b[i], sum, a[i][i], x[i]);
	}

	//scanf ("%c", &any);
    /* Print result */
	for (i = 0; i < N; i++) {
	    //printf ("%f \t %f \t %f\n", b[i], x[i], a[i][i]);
		printf ("%f \n", x[i]);
	}

}
