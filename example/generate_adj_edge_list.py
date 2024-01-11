import torch
from torch_graphgen import LayerGraph

# import pickle as pkl

model = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", pretrained=True)
model.eval()

graph = LayerGraph(model)
graph.to_component_adj_list(output="inception_v3.adjlist")
graph.to_component_edge_list(output="inception_v3.edgelist")
# with open("model.adjlist", "w") as f:
#     for edge in graph.adj_list:
#         # if len(line) == 0:
#         #     continue
#         string = " ".join(map(str, edge))
#         f.write(string + "\n")
#
# with open("model.edgelist", "w") as f:
#     for edge in graph.edge_list[:-1]:
#         string = " ".join(map(str, edge))
#         f.write(string + "\n")
#     string = " ".join(map(str, graph.edge_list[-1]))
#     f.write(string)
