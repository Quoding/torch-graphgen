import torch
from torch_graphgen import LayerGraph
from pprint import pprint

model = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", pretrained=True)
model.eval()

graph = LayerGraph(model)
print(graph.graph)
