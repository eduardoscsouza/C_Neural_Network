all:
	gcc neuralnetwork.c main.c -fopenmp -lm -O3 -o neuralnetwork.out 

clean:
	rm *.o *.out

run:
	./multmatrix.out

test:
	gcc neuralnetwork.c main.c -fopenmp -lm -Wall -Wextra -Wno-unused-parameter -g -o neuralnetwork.out
	valgrind --leak-check=full --track-origins=yes ./neuralnetwork.out