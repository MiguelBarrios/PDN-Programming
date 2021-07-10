all: a b c map

a: kmeans_parallel_a.c
	gcc -fopenmp -g -Wall -o kmeans_parallel_a kmeans_parallel_a.c -lm
	#gcc -Xpreprocessor -fopenmp -g -Wall -o kmeans_parallel_a kmeans_parallel_a.c -lomp

b: kmeans_parallel_b.c
	gcc -fopenmp -g -Wall -o kmeans_parallel_b kmeans_parallel_b.c -lm
	#gcc -Xpreprocessor -fopenmp -g -Wall -o kmeans_parallel_b kmeans_parallel_b.c -lomp

c: kmeans_parallel_c.c
	gcc -fopenmp -g -Wall -o kmeans_parallel_c kmeans_parallel_c.c -lm
	#gcc -Xpreprocessor -fopenmp -g -Wall -o kmeans_parallel_c kmeans_parallel_c.c -lomp


map: kmeans_parallel_map.c
	gcc -fopenmp -g -Wall -o kmeans_parallel_map kmeans_parallel_map.c -lm
	#gcc -Xpreprocessor -fopenmp -g -Wall -o kmeans_parallel_map kmeans_parallel_map.c -lomp

runb:
	./kmeans_parallel_b 4000000 points_4M.csv 2 16 initial_centroid.csv final_a_1M_th1_centroid_values.csv time_a_1M_th1.csv

runa:
	./kmeans_parallel_a 4000000 points_4M.csv 2 16 initial_centroid.csv final_a_1M_th1_centroid_values.csv time_a_1M_th1.csv

runc:
	./kmeans_parallel_c 4000000 points_4M.csv 2 16 initial_centroid.csv final_a_1M_th1_centroid_values.csv time_a_1M_th1.csv

run:
	./kmeans_parallel_map 4000000 points_4M.csv 2 16 initial_centroid.csv final_a_1M_th1_centroid_values.csv time_a_1M_th1.csv


clean:
	$(RM) kmeans_parallel_a kmeans_parallel_b kmeans_parallel_c

#b: kmeans_parallel_b.c
#	gcc -fopenmp -g -Wall -o kmeans_parallel_b kmeans_parallel_b.c -lm

#c: kmeans_parallel_c.c
#	gcc -fopenmp -g -Wall -o kmeans_parallel_c kmeans_parallel_c.c -lm

