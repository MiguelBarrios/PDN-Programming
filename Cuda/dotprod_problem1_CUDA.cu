#include <stdio.h>
#include <stdlib.h> // for strtol
#include <string.h>
#include <time.h>
#include "support.h"
#include "kernel.cu"

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

    vec_1 = (float*) malloc(vec_size*sizeof(float));
    vec_2 = (float*) malloc(vec_size*sizeof(float));

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

    //pointers to gpu
    float *d_A, *d_B, *d_C;

    //size of arrays
    int size = vec_size*sizeof(float);

    // allocate memory for input on the device -----------------------------------------
    if( cudaMalloc((void**) &d_A, size) != cudaSuccess) {
        FATAL("Unable to allocate device memory for d_A");
    }

    if( cudaMalloc((void**) &d_B, size)!= cudaSuccess ){
        FATAL("Unable to allocate device memory for d_B");
    }

    if( cudaMalloc((void**) &d_C, size) != cudaSuccess){
        FATAL("Unable to allocate device memory for d_C");
    }


    clock_gettime(CLOCK_REALTIME, &start);
    // 1. Transfer the A vector to the GPU global memory ------------------------------------------------------------
    if( cudaMemcpy(d_A, vec_1, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        FATAL("Unable to copy vec1 to device");
    }
    
    // 1. Transfer the B vector to the GPU global memory ------------------------------------------------------------
    if( cudaMemcpy(d_B, vec_2, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        FATAL("Unable to copy vec2 to device");
    }

    clock_gettime(CLOCK_REALTIME, &end);

    double time_spent_1 = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;


    cudaDeviceSynchronize();

    // Launche kernel --------------------------------------------------------------------

    printf("Launching kernel..."); 
    fflush(stdout);

    const unsigned int THREADS_PER_BLOCK = 512;
    const unsigned int numBlocks = (vec_size - 1)/THREADS_PER_BLOCK + 1;
    dim3 gridDim(numBlocks, 1, 1);
    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);

    // 3. Run the kernel function to compute the C vector by element-wise multiplication
    clock_gettime(CLOCK_REALTIME, &start);
    
    dotProductMultiplicationPart<<< gridDim, blockDim >>> (d_A, d_B, d_C, vec_size);

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &end);

    double time_spent_2 = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;

    // 4. Transfer the C vector from the GPU global memory to the system memory
    clock_gettime(CLOCK_REALTIME, &start);               
    
    cudaMemcpy( vec_1, d_C, size, cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_REALTIME, &end);

    double time_spent_3 = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;



    // 5. Compute the dot product on the CPU
    clock_gettime(CLOCK_REALTIME, &start); 
    int i;
    for (i = 0; i<vec_size; i++)    
    {
        dot_product += vec_1[i];
    }
    clock_gettime(CLOCK_REALTIME, &end);
    double time_spent_4 = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    fprintf(outputFile, "%f", dot_product);
    fprintf(outputFile2, "%.20f", time_spent_1);
    fprintf(outputFile2, "%.20f", time_spent_2);
    fprintf(outputFile2, "%.20f", time_spent_3);
    fprintf(outputFile2, "%.20f", time_spent_4);

    fclose (outputFile);
    fclose (outputFile2);

    free(vec_1);
    free(vec_2);

    return 0;

}

