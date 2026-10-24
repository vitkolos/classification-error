import torch

import classerr.neurons
import classerr.utils


def build_tree(model: torch.nn.Module, dataset: classerr.utils.Dataset):
    constraints = classerr.neurons.Constraints()
    message = classerr.neurons.Message(dataset[0][0])
    message = classerr.neurons.collect(model.layers, message, constraints)
    num_neurons = sum(map(len, constraints.neuron_states))
    breakpoint()
