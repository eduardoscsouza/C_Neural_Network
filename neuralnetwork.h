#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H



typedef unsigned short int nn_size_t;
typedef float nn_float_t;

typedef struct Neuron
{
	nn_size_t n_dim;
	nn_float_t * weights;
	nn_float_t (*actv)(nn_float_t);
}Neuron;

typedef struct Layer
{
	nn_size_t n_neurons, in_size;
	Neuron * neurons;
	nn_float_t (*actv)(nn_float_t);
}Layer;

typedef struct Network
{
	nn_size_t n_layers, in_size;
	Layer * layers;
}Network;

Neuron * new_neuron();
void delete_neuron(Neuron *);
nn_float_t actv_neuron(Neuron *);

Layer * new_layer();
void delete_layer(Layer *);
nn_float_t * actv_layer(Layer *);

Network * new_network();
void delete_network(Network *);
nn_float_t * actv_network(Network *);



#endif