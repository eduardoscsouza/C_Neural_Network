#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "neuralnetwork.h"

#define TEST_NEURON_N_DIM 5

#define TEST_LAYER_N_NEURONS 3
#define TEST_LAYER_IN_SIZE 3

#define TEST_NETWORK_N_LAYERS 5
#define TEST_NETWORK_LAYERS_SIZES {5, 5, 3, 2, 1}
#define TEST_NETWORK_LAYERS_ACTVS {&relu, &sigm, &relu, &sigm, &sigm}
#define TEST_NETWORK_IN_SIZE 6

#define TEST_XOR_N_LAYERS 2
#define TEST_XOR_LAYERS_SIZES {2, 1}
#define TEST_XOR_LAYERS_ACTVS {&step, &step}
#define TEST_XOR_IN_SIZE 2



nn_float_t * rand_vect(int n)
{
	nn_float_t * vect = (nn_float_t*) malloc(n*sizeof(nn_float_t));

	int i;
	for (i=0; i<n; i++) vect[i] = rand() / (nn_float_t)RAND_MAX;

	return vect;
}

void print_vect(nn_float_t * vect, int n)
{
	int i;
	for (i=0; i<n; i++) printf("%f; ", vect[i]);
	printf("\n");
}



int main(int argc, char * argv[])
{
	Neuron * neuron = new_neuron(TEST_NEURON_N_DIM, &linear);
	srand(time(NULL));
	nn_float_t * in = rand_vect(neuron->n_dim);
	
	printf("Input: ");
	print_vect(in, neuron->n_dim);
	print_neuron(neuron);
	nn_float_t net = neuron_forward(neuron, in);
	printf("Net: %f\nf(net): %f\n", net, sigm(net));
	delete_neuron(neuron);
	free(in);
	printf("\n\n");



	Layer * layer = new_layer(TEST_LAYER_N_NEURONS, TEST_LAYER_IN_SIZE, &sigm);
	in = rand_vect(layer->in_size);
	printf("Input: ");
	print_vect(in, layer->in_size);
	print_layer(layer);
	nn_float_t * out = layer_forward(layer, in);
	printf("Output: ");
	print_vect(out, layer->n_neurons);
	delete_layer(layer);
	free(in);
	free(out);
	printf("\n\n");



	nn_size_t aux_layers_sizes[] = TEST_NETWORK_LAYERS_SIZES;
	nn_float_t (*aux_layers_actvs[])(nn_float_t) = TEST_NETWORK_LAYERS_ACTVS;
	Network * network = new_network(TEST_NETWORK_N_LAYERS, aux_layers_sizes, aux_layers_actvs, TEST_NETWORK_IN_SIZE);
	in = rand_vect(network->in_size);
	printf("Input: ");
	print_vect(in, network->in_size);
	print_network(network);
	out = network_forward(network, in);
	printf("Output: ");
	print_vect(out, network->layers[network->n_layers-1]->n_neurons);
	delete_network(network);
	free(in);
	free(out);
	printf("\n\n");



	nn_size_t xor_layers_sizes[] = TEST_XOR_LAYERS_SIZES;
	nn_float_t (*xor_layers_actvs[])(nn_float_t) = TEST_XOR_LAYERS_ACTVS;
	network = new_network(TEST_XOR_N_LAYERS, xor_layers_sizes, xor_layers_actvs, TEST_XOR_IN_SIZE);
	
	network->layers[0]->neurons[0]->weights[0] = network->layers[0]->neurons[0]->weights[1] = 1.0;
	network->layers[0]->neurons[0]->weights[2] = -0.5;
	network->layers[0]->neurons[1]->weights[0] = network->layers[0]->neurons[1]->weights[1] = -1.0;
	network->layers[0]->neurons[1]->weights[2] = 1.5;
	network->layers[1]->neurons[0]->weights[0] = network->layers[1]->neurons[0]->weights[1] = 1.0;
	network->layers[1]->neurons[0]->weights[2] = -1.5;
	print_network(network);

	in = rand_vect(network->in_size);
	int i;
	for (i=0; i<4; i++){
		in[0] = (nn_float_t)(i&1);
		in[1] = (nn_float_t)((i&2)>>1);
		printf("Input: ");
		print_vect(in, network->in_size);
		
		out = network_forward(network, in);
		printf("Output: ");
		print_vect(out, network->layers[network->n_layers-1]->n_neurons);
		free(out);
	}

	delete_network(network);
	free(in);



	return 0;
}