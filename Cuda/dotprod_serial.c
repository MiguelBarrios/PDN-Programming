#include <stdio.h>
#include <stdlib.h> // for strtol
#include <string.h>
#include <time.h>

#define MAXCHAR 25
#define BILLION  1000000000.0

int main (int argc, char *argv[])
{
    if( argc != 6)
    {
        printf("USE LIKE THIS: dotprod_serial vector_size vec_1.csv vec_2.csv result.csv time.csv\n");
        return EXIT_FAILURE;
    }

    int vec_size;
    vec_size = strtol(argv[1], NULL, 10);
    // printf("n_points = %d\n", n_points);

    FILE *inputFile1;

    inputFile1 = fopen(argv[2], "r");
    if (inputFile1 == NULL){
        printf("Could not open file %s",argv[2]);
    }

    FILE *inputFile2;

    inputFile2 = fopen(argv[3], "r");
    if (inputFile2 == NULL){
        printf("Could not open file %s",argv[3]);
    }

    FILE *outputFile;
    outputFile = fopen(argv[4], "w");

    FILE *outputFile2;
    outputFile2 = fopen(argv[5], "w");

    float* vec_1;
    float* vec_2;

    char str[MAXCHAR];

    vec_1 = malloc(vec_size*sizeof(float));
    vec_2 = malloc(vec_size*sizeof(float));

    // Store values of vector 1
    int k = 0;
    while (fgets(str, MAXCHAR, inputFile1) != NULL)
    {
        sscanf( str, "%f", &(vec_1[k]) );
        k++;
    }
    fclose(inputFile1); 

    // Store values of vector 2
    k = 0;

    while (fgets(str, MAXCHAR, inputFile2) != NULL)
    {
        sscanf( str, "%f", &(vec_2[k]) );
        k++;
    }

    fclose(inputFile2); 
  
    float dot_product = 0.0;

    // infor about timing in C
    // https://www.techiedelight.com/find-execution-time-c-program/
    struct timespec start, end;
    
    clock_gettime(CLOCK_REALTIME, &start);
    
	// Performing for product
    int i;
    for (i = 0; i<vec_size; i++)    
    {
        dot_product += vec_1[i]*vec_2[i];
    }

    clock_gettime(CLOCK_REALTIME, &end);
    double time_spent = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;


    fprintf(outputFile, "%f", dot_product);
    fprintf(outputFile2, "%.20f", time_spent);

    fclose (outputFile);
    fclose (outputFile2);

    free(vec_1);
    free(vec_2);

    return 0;
}