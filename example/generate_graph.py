import torch
from src.utils import generate_graph_from_model


model = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", pretrained=True)
model.eval()
graph = generate_graph_from_model(model)
print(graph.graph)
