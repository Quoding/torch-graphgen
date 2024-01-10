import torch
from torch_graphgen import LayerGraph

# import pickle as pkl

model = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", pretrained=True)
model.eval()

graph = LayerGraph(model)
graph.gen_component_adj_list()

with open("model.adjlist", "w") as f:
    for line in graph.adj_list:
        if len(line) == 0:
            continue
        string = " ".join(map(str, line))
        f.write(string + "\n")
