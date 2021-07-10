#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

int main(int argc, char *argv[]) {

   int        comm_sz;               /* Number of processes    */
   int        my_rank;               /* My process rank        */
   double t1, t2;

   //printf("Num Args: %d %s\n", argc, argv[1]);


   int arr_size = atoi(argv[1]);
   int my_array[arr_size];

   printf("Array size: %d\n", arr_size);
   /* Start up MPI */
   MPI_Init(NULL, NULL); 

   /* Get the number of processes */
   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); 

   /* Get my rank among all the processes */
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 

   // Create array of random numbers
   if (my_rank == 0)
   {
      int i;
      for(i = 0; i < arr_size; ++i)
      {
         my_array[i] = rand();
      }
   }

   t1 = MPI_Wtime();

   int count = 0;
   while(count < 1000)
   {

      if(my_rank == 0)
      {
         //Send array to process 1
         MPI_Send(my_array, arr_size, MPI_INT, 1, 0,MPI_COMM_WORLD);

         //Recive Arary from process 1
         MPI_Recv(my_array, arr_size, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      else if(my_rank == 1)
      {
         //Recive Array from process 0
         MPI_Recv(my_array, arr_size, MPI_INT, 0,0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

         // Send Array to process 1
         MPI_Send(my_array, arr_size, MPI_INT, 0,0, MPI_COMM_WORLD);
      }
      ++count;

   }

   if(my_rank == 0)
   {
      t2 = MPI_Wtime();
      double elapsed = t2 - t1;
      printf("elapsed time = %f\n", elapsed);
      FILE *outputFile;
      outputFile = fopen(argv[2], "w");
      fprintf(outputFile, "%f", elapsed);
      fclose (outputFile);
   }


   /* Shut down MPI */
   MPI_Finalize(); 

   return 0;
}  /* main */
