all: dotprod_MPI.c
	mpicc -g -Wall -o dotprod_MPI dotprod_MPI.c

clean:
	$(RM) dotprod_MPI

run:
	mpirun -n 2 ./dotprod_MPI 1048576 vec1_2_20.csv vec2_2_20.csv result_prob3_MPI_p2_s1.csv time_prob3_MPI_p2_s1.csv
