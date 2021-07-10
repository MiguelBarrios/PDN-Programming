#include <stdio.h>
#include <stdlib.h> // for strtol
#include <string.h>
#include <omp.h>
#include <math.h> // pow
#include <float.h>

#define MAXCHAR 25

double calc_distance(double x1, double x2, double y1, double y2)
{
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

double findClosesGroup(double* centroidX, double* centroidY,double x1,double y1, int numCentroids)
{
    int bestGroup = 0;
    double distanceToBestGroup = DBL_MAX;
    int i;
    for(i = 0; i < numCentroids; ++i)
    {
        double x2 = centroidX[i];
        double y2 = centroidY[i];
        double distance = calc_distance(x1,x2,y1,y2);
        if(distance < distanceToBestGroup)
        {
            distanceToBestGroup = distance;
            bestGroup = i;
        }
        //printf("\tGroup: %d distance %f\n", i, distance);
    }
    //printf("(%f,%f) bestGroup %d distance %f\n", x1,y1, bestGroup, distanceToBestGroup);
    return bestGroup;
}

void displayArr(double* x, double* y, int size)
{
    int row;
    for(row = 0; row < size; ++row)
    {
        printf("%f %f\n", x[row], y[row]);
    }
}

int main (int argc, char *argv[])
{
    if( argc != 8)
    {
        printf("USE LIKE THIS: kmeans_parallel n_points points.csv num_threads n_centroids initial_centroid_values.csv final_centroid_values.csv time.csv\n");
        return EXIT_FAILURE;
    }

    int n_points;
    n_points = strtol(argv[1], NULL, 10);

    double* array_x;
    double* array_y;

    double* array_centroid_x;
    double* array_centroid_y;

    // array that contains the centroid which the point belongs too
    int* group = malloc(n_points*sizeof(int));

    //int* local_array_x;
    //int* local_array_y;

    FILE *inputFile1;
    FILE *inputFile2;

    //char* p1;
    //char* p2;

    inputFile1 = fopen(argv[2], "r");
    if (inputFile1 == NULL){
        printf("Could not open file %s",argv[2]);
    }

    int thread_count;
    thread_count = strtol(argv[3], NULL, 10);

    int n_centroids;
    n_centroids = strtol(argv[4], NULL, 10);

    inputFile2 = fopen(argv[5], "r");
    if (inputFile2 == NULL){
        printf("Could not open file %s",argv[5]);
    }

    FILE *outputFile;
    outputFile = fopen(argv[6], "w");

    FILE *outputFile2;
    outputFile2 = fopen(argv[7], "w");

    char str[MAXCHAR];

    array_x = malloc(n_points*sizeof(double));
    array_y = malloc(n_points*sizeof(double));

    array_centroid_x = malloc(n_centroids*sizeof(double));
    array_centroid_y = malloc(n_centroids*sizeof(double));

    //printf("Before storing points\n");

    // Store values of 2D points
    int k = 0;
    while (fgets(str, MAXCHAR, inputFile1) != NULL)
    {
        sscanf( str, "%lf,%lf", &(array_x[k]), &(array_y[k]) );
        //printf("i: %d, x: %lf, y: %lf\n", k,  array_x[k], array_y[k]);
        k++;
    }
    fclose(inputFile1); 

    // Storing centroid values
    k = 0;

    while (fgets(str, MAXCHAR, inputFile2) != NULL)
    {
        sscanf( str, "%lf,%lf", &(array_centroid_x[k]), &(array_centroid_y[k]) );
        //printf("i: %d, x: %lf, y: %lf\n", k,  array_centroid_x[k], array_centroid_y[k]);
        k++;
    }

    fclose(inputFile2); 

    printf("thread_count = %d\n", thread_count);

    double start = omp_get_wtime();

    double avg_distance_moved = 2;
    while(avg_distance_moved > 1)
    {
        // Assign each point to the nearest centroid

        #pragma omp parallel shared(array_centroid_x,array_centroid_y,array_x,array_y, n_centroids, group)
        {
            int i;
            # pragma omp parallel for  num_threads(thread_count)
            for(i = 0; i < n_points; ++i){
                group[i] = findClosesGroup(array_centroid_x, array_centroid_y,array_x[i],array_y[i], n_centroids);
            }

            //Compute new location for each centroid
            int centroidSumX [16]  = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
            int centroidSumY [16]  = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
            int centroidCount [16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            // Calc new location for each centroid
            # pragma omp parallel for 
            for(i = 0; i < n_points; ++i)
            {
                double x = array_x[i];
                double y = array_y[i];
                int bestGroup = group[i];

                # pragma omp atomic
                centroidSumX[bestGroup] = x + centroidSumX[bestGroup];
                            
                # pragma omp atomic
                centroidSumY[bestGroup] = y + centroidSumY[bestGroup];

                # pragma omp atomic
                centroidCount[bestGroup] = centroidCount[bestGroup] + 1;



            }

            double total_distance_moved = 0;
            for(i = 0; i < n_centroids; ++i)
            {
                double new_x = centroidSumX[i] / centroidCount[i];
                double new_y = centroidSumY[i] / centroidCount[i];   

                double distanceMoved = calc_distance(array_centroid_x[i], new_x, array_centroid_y[i], new_y);
                total_distance_moved += distanceMoved;

                //assign new coordinates for centroids
                //printf("Centroid %d\n", i);
                //printf("old x = %.2f y = %.2f new x = %.2f y %.2f\n", array_centroid_x[i], array_centroid_y[i], new_x, new_y);
                //printf("Distance Moved = %f\n\n", distanceMoved);
                array_centroid_x[i] = new_x;
                array_centroid_y[i] = new_y;
            }
            avg_distance_moved = total_distance_moved / n_centroids;
            //printf("Avg centroid moving distance = %f\n", avg_distance_moved);

        }
    }

    double end = omp_get_wtime();


    
    // Time calculation (in seconds)

    double time_passed = end - start;

    printf("Time passed %f num_threads %d\n", time_passed, thread_count);

    fprintf(outputFile2, "%f", time_passed);

    int i;
    for(i = 0; i < n_centroids; ++i)
    {
        fprintf(outputFile, "%f,%f\n", array_centroid_x[i], array_centroid_y[i]);
    }

    fclose (outputFile);
    fclose (outputFile2);

    free(array_x);
    free(array_y);
    free(array_centroid_x);
    free(array_centroid_y);
    printf("End of Program\n");

    return 0;
}