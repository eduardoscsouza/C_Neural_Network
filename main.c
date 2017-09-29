#include <stdio.h>
#include <stdlib.h>

#include "neuralnetwork.h"

#define TEST_N_DIM 5



int main(int argc, char * argv[])
{

	Neuron * neuron = new_neuron(TEST_N_DIM, &linear);
	nn_float_t * in = (nn_float_t*) malloc(neuron->n_dim*sizeof(nn_float_t));
	int i;
	for (i=0; i<neuron->n_dim; i++) in[i] = rand() / (nn_float_t)RAND_MAX;
	
	for(i=0; i<neuron->n_dim+1; i++) printf("%f ", neuron->weights[i]);
	printf("\n");
	for(i=0; i<neuron->n_dim; i++) printf("%f ", in[i]);
	printf("\n");
	
	nn_float_t net = neuron_forward(neuron, in);
	printf("%f\n%f\n\n", net, sigm(net));
	delete_neuron(neuron);
	free(in);


	return 0;
}