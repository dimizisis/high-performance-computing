#include <stdio.h>
#include "mpi.h"

#define N 12
#define ROOT 0
#define TERMINATE -1
#define TAG 100

void display_time(double start, double end);
 
int main(int argc, char *argv[]){
    int rank, size;
    MPI_Status status;
    double begin, end;

    MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == ROOT){
        begin = MPI_Wtime();
        int i, x=2;
        (void) printf("%d ", x);    /* Print the first prime */
        for (i=3; i<=size; ++i)
            if (i % x != 0)
                MPI_Send(&i, 1, MPI_INT, rank+1, TAG, MPI_COMM_WORLD);  /* Send primes to P1 */

        i = TERMINATE;
        MPI_Send(&i, 1, MPI_INT, rank+1, TAG, MPI_COMM_WORLD);  /* Send termination signal */
    }

    else {
        int x, i=0;
        MPI_Recv(&x, 1, MPI_INT, rank-1, TAG, MPI_COMM_WORLD, &status);   /* Receive a prime number and print it */
        (void) printf("%d ", x);
        while(i != TERMINATE){ 
            MPI_Recv(&i, 1, MPI_INT, rank-1, TAG, MPI_COMM_WORLD, &status); /* Keep receiving numbers until termination signal is received */
            if (i != TERMINATE && !(i % x)) /* If termination signal not received & i is prime */
                if (rank != size-1)
                    MPI_Send(&i, 1, MPI_INT, rank+1, TAG, MPI_COMM_WORLD);  /* Send it if it is not the last process */
                else
                    (void) printf("%d ", i);    /* Print it if it is the last process */
        }
    }

    if (rank == ROOT){
        end = MPI_Wtime();
        display_time(begin, end);   /* runtime */
    }
    
    MPI_Finalize();

    return 0;
}

/*
 * Function:  display_time 
 * --------------------
 * Prints the time (seconds) elapsed
 *
 *  start: the time (seconds) in which the whole process started
 *  end: the time (seconds) in which the whole process started
 *
 */

void display_time(double start, double end){
    (void) printf("Time spent for sorting: %g seconds\n\n", (double)(end-start));
}
