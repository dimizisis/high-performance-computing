#include <stdio.h>
#include "mpi.h"

#define ROOT 0
#define TERMINATE -1
#define TAG 1000
#define N 10000

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
        (void) printf("%d %d\n", rank, x);    /* Print the first prime */
        for (i=3; i<=N; ++i)
            if (i % x != 0)
                MPI_Send(&i, 1, MPI_INT, rank+1, TAG, MPI_COMM_WORLD);  /* Send primes to P1 */

        i = TERMINATE;
        MPI_Send(&i, 1, MPI_INT, rank+1, TAG, MPI_COMM_WORLD);  /* Send termination signal */
        MPI_Recv(&i, 1, MPI_INT, size-1, TAG, MPI_COMM_WORLD, &status); 
        end = MPI_Wtime();
        display_time(begin, end);   /* runtime */
    }

    if (rank == (size-1)){
        int i, x=0;
        while(1){ 
            MPI_Recv(&i, 1, MPI_INT, rank-1, TAG, MPI_COMM_WORLD, &status);
            if (x == 0) x = i;/* Keep receiving numbers until termination signal is received */
            if (i == TERMINATE) {
            	MPI_Send(&i, 1, MPI_INT, 0, TAG, MPI_COMM_WORLD);
                break;  /* Send it if it is not the last process */   
            }        
        }
        (void) printf("%d %d\n", rank, x);    /* Print it if it is the last process */
    }
    
    if ((rank != ROOT) && (rank !=(size-1))){
        int i, x=0;
        while(1){ 
            MPI_Recv(&i, 1, MPI_INT, rank-1, TAG, MPI_COMM_WORLD, &status);
            if (x == 0) x = i;
            if (i % x != 0 || i == TERMINATE) /* If termination signal received OR i is prime */
            	MPI_Send(&i, 1, MPI_INT, rank+1, TAG, MPI_COMM_WORLD);
            if (i == TERMINATE) 
                break;  /* Send it if it is not the last process */           
        }
        (void) printf("%d %d\n", rank, x);    /* Print it if it is the last process */
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
