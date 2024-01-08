import torch
from torch_graphgen import model_to_layer_graph, model_to_neuron_graph


model = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", pretrained=True)
model.eval()
graph = model_to_layer_graph(model)
graph = model_to_neuron_graph(model)
print(graph.graph)
