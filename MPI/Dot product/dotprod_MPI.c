#include <stdio.h>
#include <stdlib.h> // for strtol
#include <string.h>
#include <time.h>
#include <mpi.h>

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
    // printf("n_points = %d\n", vec_size);


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


    int comm_sz; // Number of processes
    int my_rank; // My process rank

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int section_size = vec_size / comm_sz;

    //read in files
    if(my_rank == 0)
    {
        char str[MAXCHAR];

        vec_1 = malloc(vec_size*sizeof(float));
        vec_2 = malloc(vec_size*sizeof(float));

        // printf("Store values vec1\n");
        // Store values of vector
        int k = 0;
        while (fgets(str, MAXCHAR, inputFile1) != NULL)
        {
            sscanf( str, "%f", &(vec_1[k]) );
            k++;
        }
        fclose(inputFile1); 

        // printf("Store values vec2/n");
        // Store values of vector
        k = 0;

        while (fgets(str, MAXCHAR, inputFile2) != NULL)
        {
            sscanf( str, "%f", &(vec_2[k]) );
            k++;
        }

        fclose(inputFile2); 
    }


    if(my_rank != 0)
    {
        vec_1 = malloc(section_size*sizeof(float));
        vec_2 = malloc(section_size*sizeof(float));

        //Recive section of array proccess will be working on
        MPI_Recv(vec_1, section_size, MPI_FLOAT, 0,0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(vec_2, section_size, MPI_FLOAT, 0,0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        float partial_dot_prod = 0;
        int i;
        for(i = 0; i < section_size; ++i)
        {
            partial_dot_prod += vec_1[i] * vec_2[i];
        }

        //printf("PID: %d startIndex: %d endIndex: %d\n",my_rank, startIndex, endIndex);
        //printf("Partial sum: %f\n", partial_dot_prod);

        // Send partial dot prod results to process 0
        float *ptr;
        ptr = &partial_dot_prod;
        MPI_Send(ptr, 1, MPI_FLOAT, 0,0, MPI_COMM_WORLD);
    }
    else if(my_rank == 0)
    {

        struct timespec start, end;
    
        clock_gettime(CLOCK_REALTIME, &start);

        int pid;
        for(pid = 1; pid < comm_sz; ++pid)
        {
            // Send vector array to other processes
            MPI_Send(&vec_1[pid * section_size], section_size, MPI_FLOAT, pid, 0, MPI_COMM_WORLD);
            MPI_Send(&vec_2[pid * section_size], section_size, MPI_FLOAT, pid, 0, MPI_COMM_WORLD);
        }


        //printf("Number of processes: %d\n\n", comm_sz);
        int startIndex = my_rank * section_size;
        int endIndex = (startIndex + section_size) - 1;

        float dot_product = 0;

        int i;
        for(i = startIndex; i < endIndex; ++i)
        {
            dot_product += vec_1[i] * vec_2[i];
        }

        //printf("PID: %d startIndex: %d endIndex: %d\n", my_rank, startIndex, endIndex);

        float *ptr;
        float partial = 0;
        ptr = &partial;

        for(pid = 1; pid < comm_sz; ++pid)
        {
            MPI_Recv(ptr, 1, MPI_FLOAT, pid, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            dot_product += *ptr;
        }

        //printf("Dot Product: %f\n", dot_product);


        clock_gettime(CLOCK_REALTIME, &end);
        double time_spent = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;

        //printf("Dot product: %f\n", dot_product);

        fprintf(outputFile, "%f", dot_product);
        fprintf(outputFile2, "%.20f", time_spent);

    }

    free(vec_1);
    free(vec_2);
    fclose (outputFile);
    fclose (outputFile2);




    MPI_Finalize();
    return 0;
}