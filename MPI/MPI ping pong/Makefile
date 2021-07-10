all: pingpong_MPI.c
	mpicc -g -Wall -o pingpong_MPI pingpong_MPI.c

clean:
	$(RM) pingpong_MPI

run:
	mpirun -n 2 ./pingpong_MPI 1000000 time_1M_same.csv
	#mpirun -n 2 ./pingpong_MPI 1000000 time_1M_same.csv