#include <stdio.h> 
#include <stdlib.h> 
#include <omp.h>
#define N 128
#define base 0

void zeros(int array[], int n);
void count_characters(int freq[], char buffer[], long file_size);
void display_count(int freq[], int n);

int main (int argc, char *argv[]) {
	
	FILE *pFile;
	long file_size;
	char * buffer;
	char * filename;
	size_t result;
	int freq[N];
	double start, end;

    if (argc != 2) {
		printf ("Usage : %s <file_name>\n", argv[0]);
		return 1;
    }
	filename = argv[1];
	pFile = fopen ( filename , "rb" );
	if (pFile==NULL) {printf ("File error\n"); return 2;}

	// obtain file size:
	fseek (pFile , 0 , SEEK_END);
	file_size = ftell (pFile);
	rewind (pFile);
	printf("file size is %ld\n", file_size);
	
	// allocate memory to contain the file:
	buffer = (char*) malloc (sizeof(char)*file_size);
	if (buffer == NULL) {printf ("Memory error\n"); return 3;}

	// copy the file into the buffer:
	result = fread (buffer,1,file_size,pFile);
	if (result != file_size) {printf ("Reading error\n"); return 4;} 

	zeros(freq, N);

    start = omp_get_wtime();

	count_characters(freq, buffer, file_size);

	end = omp_get_wtime();

	display_count(freq, N);	

	(void) printf("Time spent for counting: %g", (end - start));

	fclose (pFile);
	free (buffer);

	return 0;
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

void zeros(int array[], int n){
	int j;
	for (j=0; j<n; ++j)
		array[j]=0;
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
	int i,j;
	int local_freq[N];

	#pragma omp parallel default(shared) private(i, j, local_freq)
	{
		zeros(local_freq, N);
		
		#pragma omp for
		for (i=0; i<file_size; ++i)
			++local_freq[buffer[i] - base];
		
		for (j=0; j<N; ++j)
			#pragma omp critical
			freq[j] += local_freq[j];
	}
		
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