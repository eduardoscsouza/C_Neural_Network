#include "neuralnetwork.h"

#define _GNU_SOURCE
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <limits.h>

#define NEURON_PARALLEL 0
#define NEURON_N_THREADS 8
#define LAYER_PARALLEL 1
#define LAYER_N_THREADS 8

#define INIT_MAX 1.0
#define INIT_MIN -1.0



Neuron * new_neuron(nn_size_t n_dim, nn_float_t (*actv)(nn_float_t))
{
	Neuron * neuron = (Neuron*) malloc(sizeof(Neuron));
	
	neuron->actv = actv;
	neuron->n_dim = n_dim;
	
	neuron->weights = (nn_float_t*) malloc((n_dim+1) * sizeof(nn_float_t));
	unsigned long long aux_rand;
	nn_size_t i;
	for (i=0; i<n_dim+1; i++){
		syscall(SYS_getrandom, &aux_rand, sizeof(aux_rand), 0);
		neuron->weights[i] = ((aux_rand / (nn_float_t)ULLONG_MAX)*(INIT_MAX-INIT_MIN)) + INIT_MIN;
	}

	return neuron;
}

void delete_neuron(Neuron * neuron)
{
	free(neuron->weights);
	free(neuron);
}

nn_float_t neuron_forward(Neuron * neuron, nn_float_t * input)
{
	nn_float_t net = 0.0;

	#if NEURON_PARALLEL
		#pragma omp parallel num_threads(NEURON_N_THREADS)
		{
			int id = omp_get_thread_num();
			nn_size_t block_size = neuron->n_dim/omp_get_num_threads(), lower_bound = block_size*id, upper_bound = block_size*(id+1);
			if (id == omp_get_max_threads()-1) upper_bound = neuron->n_dim;

			nn_float_t aux;
			nn_size_t i;
			for(i=lower_bound; i<upper_bound; i++){
				aux = neuron->weights[i] * input[i];
				#pragma omp critical(neuron_sum)
				{
					net += aux;
				}
			}
		}
		net += neuron->weights[neuron->n_dim];
	#else
		nn_size_t i;
		for (i=0; i<neuron->n_dim; i++) net += neuron->weights[i] * input[i];
		net += neuron->weights[neuron->n_dim];
	#endif

	return neuron->actv(net);
}



Layer * new_layer()
{
	return NULL;
}

void delete_layer(Layer * layer)
{

}

nn_float_t * layer_forward(Layer * layer)
{
	return NULL;
}



Network * new_network()
{
	return NULL;
}

void delete_network(Network * network)
{

}

nn_float_t * network_forward(Network * network)
{
	return NULL;
}



nn_float_t relu(nn_float_t net)
{
	if (net>=0.0) return net;
	else return 0.0;
}

nn_float_t soft_relu(nn_float_t net)
{
	return log(1.0 + exp(net));
}

nn_float_t step(nn_float_t net)
{
	if (net>=0.0) return 1.0;
	else return 0.0;
}

nn_float_t sigm(nn_float_t net)
{
	return 1.0/(1.0 + exp(-net));
}

nn_float_t linear(nn_float_t net)
{
	return net;
}