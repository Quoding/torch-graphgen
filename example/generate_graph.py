import torch
from torch_graphgen import LayerGraph
from pprint import pprint

model = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", pretrained=True)
model.eval()

graph = LayerGraph(model)
fake_data = torch.rand(1, 3, 299, 299)
graph.write_activation_features(fake_data, "asda.txt")
# print(graph.graph)
