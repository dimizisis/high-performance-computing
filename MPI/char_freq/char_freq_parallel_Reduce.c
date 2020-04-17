#include <stdio.h> 
#include <stdlib.h> 
#include "mpi.h"

#define NUM_OF_ARGS 2
#define ROOT 0
#define N 128
#define base 0

int check_args(int argc);
FILE* open_file(char* filename);
long obtain_file_size(FILE* pFile);
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

    /* Check if the right number of arguments give */
    (void) check_args(argc);

    /* Open file with given filename (argv) */
    pFile = open_file(argv[1]);

    /* allocate memory to contain the file	*/
    total_freq = (int*) calloc(sizeof(int), N);
    if (total_freq == NULL) {printf ("Memory error\n"); return 3;}

    /* Get file size */
    file_size = obtain_file_size(pFile);

    if (rank == ROOT){
    	(void) printf("File size is %ld\n", file_size);
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

    /* Seek from the beggining of start	*/
    fseek(pFile, start, SEEK_SET);	

    /* Read from local buffer */
    fread(buffer, 1, local_file_size, pFile);

    /* Count chars */
    count_characters(freq, buffer, local_file_size);

    if (rank == ROOT) end = MPI_Wtime();

    /* Make the reduce */
    MPI_Reduce(freq, total_freq, N, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);

    /* Printing */
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
 * Function:  check_args 
 * --------------------
 * Checks if the correct number of arguments is given
 *
 *  argc: number of arguments given by the user
 *
 */

int check_args(int argc){
    if (argc != NUM_OF_ARGS) {
        printf("%d arguments must be given\n", argc);
        exit(1);
    }
    return 1;
}

/*
 * Function:  open_file 
 * --------------------
 * Opens file, given the filename (argv)
 *
 *  filename: filename user gave as argument
 *
 */

FILE* open_file(char* filename){
    FILE* pFile = fopen(filename, "rb");
    if (pFile==NULL) {printf ("File error\n"); exit(2);}
    return pFile;
}

/*
 * Function:  obtain_file_size 
 * --------------------
 * Gets total file size using fseek, ftell functions
 *
 *  pFile: pointer to the file we opened
 *
 */

long obtain_file_size(FILE* pFile){
    fseek (pFile, 0, SEEK_END);
    long file_size = ftell(pFile);
    rewind(pFile);

    return file_size;
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
