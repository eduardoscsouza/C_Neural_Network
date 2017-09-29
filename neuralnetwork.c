#include <neuralnetwork.h>

#include <stdlib.h>
#include <omp.h>

#define N_THREADS 8

#define INIT_MAX 0.5
#define INIT_MIN -0.5



Neuron * new_neuron(nn_size_t n_dim, nn_float_t (*actv)(nn_float_t))
{
	Neuron * neuron = (Neuron*) malloc(sizeof(Neuron));
	
	neuron->actv = actv;
	neuron->n_dim = n_dim;
	neuron->weights = (nn_float_t*) malloc((n_dim+1) *  sizeof(nn_float_t));
	nn_size_t i;
	for (i=0; i<n_dim+1; i++) neuron->weights[i] = ((rand() / RAND_MAX)*(INIT_MAX-INIT_MIN)) - INIT_MIN;

	return neuron;
}

void delete_neuron(Neuron * neuron)
{
	free(neuron->weights);
	free(neuron);
}

nn_float_t neuron_forward(Neuron * neuron, nn_float_t * input)
{
	nn_float_t output = 0.0;

	nn_size_t block_size = neuron->n_dim/N_THREADS;
	omp_set_num_threads(N_THREADS);
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		nn_size_t lower_bound = block_size*id, upper_bound = block_size*(id+1);
		if (id == omp_get_max_threads()-1) upper_bound = neuron->n_dim;

		nn_float_t aux;
		nn_size_t i;
		for(i=lower_bound; i<upper_bound; i++){
			aux = neuron->weights[i] * input[i];
			#pragma omp critical(neuron_sum)
			{
				output += aux;
			}
		}
	}
}