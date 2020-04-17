#include <stdio.h> 
#include <stdlib.h> 
#include "mpi.h"

#define ROOT 0
#define N 128
#define base 0

void zeros(int array[], int n);
void count_characters(int freq[], char buffer[], long file_size);
void display_count(int freq[], int n);

int main (int argc, char *argv[]){
	
    FILE *pFile;
    long file_size;
    char * buffer;
    char * filename;
    int size, rank;
    int* total_freq;
    double begin, end;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 2) {
	printf("Usage : %s <file_name>\n", argv[0]);
	return 1;
    }

    /* allocate memory to contain the file	*/
    total_freq = (int*) calloc(sizeof(int), N);
    if (total_freq == NULL) {printf ("Memory error\n"); return 3;}

    filename = argv[1];
    pFile = fopen (filename , "rb");
    if (pFile==NULL) {printf ("File error\n"); return 2;}

    /* obtain file size	*/
    fseek (pFile, 0, SEEK_END);
    file_size = ftell(pFile);
    rewind (pFile);

    if (rank == ROOT){
    	printf("file size is %ld\n", file_size);
    	begin = MPI_Wtime();
    }

    /* These initialization will be done by all processes   */
    int* freq = (int*) calloc(sizeof(int), N);
    if (freq == NULL) {printf ("Memory error\n"); return 3;}
    int chunk = file_size / size;
    int extra = file_size % size;
    int start = rank * chunk;
    int stop = start + chunk;
    if (rank == size - 1) stop += extra;

    int local_file_size = stop - start;
	
    /* allocate memory to contain the file	*/
    buffer = (char*) malloc (sizeof(char)*local_file_size);
    if (buffer == NULL) {printf ("Memory error\n"); return 3;}

    /* seek from the beggining of start	*/
    fseek(pFile, start, SEEK_SET);	

    /* read from local buffer */
    fread(buffer, 1, local_file_size, pFile);

    /* count chars */
    count_characters(freq, buffer, local_file_size);

    if (rank == ROOT) end = MPI_Wtime();

    /* make the reduce */
    MPI_Reduce(freq, total_freq, N, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);

    if (rank == ROOT){
	display_count(total_freq, N);	
	(void) printf("Time spent for counting: %g\n", (double)(end-begin));
    }

    fclose(pFile);
    free(buffer);

    MPI_Finalize();

    return 0;
}

/*
 * Function:  count_characters 
 * --------------------
 * Counts the frequency of characters in an char array
 *
 *  freq: the array that will contain each character's frequency
 *  buffer: the array that contains the characters
 *  file_size: size of buffer
 *
 */

void count_characters(int freq[], char buffer[], long file_size){
     int i;
     for (i=0; i<file_size; ++i)
	++freq[buffer[i] - base];
}

/*
 * Function:  display count 
 * --------------------
 * Prints the array of characters' frequency
 *
 *  freq: the array of frequencies
 *  n: number of elements in the freq array
 *
 */

void display_count(int freq[], int n){
     int j;
     for (j=0; j<n; ++j)
	(void) printf("%d = %d\n", j+base, freq[j]);
}
