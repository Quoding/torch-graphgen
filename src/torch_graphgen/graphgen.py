from typing import Callable, Tuple

import torch
from torch import nn
from torch.fx.immutable_collections import immutable_list

import torch_geometric as tg

from .utils.graph import LayerGraph, LayerNode
from .utils.inclusion import is_included, search_for_next_included_layer, PROPERTY_NAMES


def model_to_layer_graph(model: nn.Module) -> LayerGraph:
    model.eval()

    computational_graph = torch.fx.symbolic_trace(model)

    layer_graph = {}

    # Create base nodes, no edges are created
    for idx, node in enumerate(computational_graph.graph.nodes):
        layer_graph[node.name] = LayerNode(node.name, node.target)

    # Create edges by iterating through the nodes
    for node in reversed(computational_graph.graph.nodes):
        cur_node = layer_graph[node.name]
        if len(node.args) == 0:
            continue
        elif type(node.args[0]) == immutable_list or type(node.args[0]) == tuple:
            for parent in node.args[0]:
                parent_node = layer_graph[parent.name]
                parent_node.children.append(cur_node.name)
                cur_node.parents.append(parent_node.name)
        else:
            parent_name = node.args[0].name
            parent_node = layer_graph[parent_name]
            parent_node.children.append(cur_node.name)
            cur_node.parents.append(parent_node.name)

    # Connect is_included layers together, clean up unimportant layers
    for node in reversed(computational_graph.graph.nodes):
        if node.op != "placeholder" and node.op != "output":
            cur_node = layer_graph[node.name]

            # Update parents to nearest included layer
            next_clean_nodes_names = []
            cur_node.parents = search_for_next_included_layer(
                model, cur_node, "parents", layer_graph, next_clean_nodes_names
            )

            # Update children to nearest included layer
            next_clean_nodes_names = []
            cur_node.children = search_for_next_included_layer(
                model, cur_node, "children", layer_graph, next_clean_nodes_names
            )

    # Clean up the graph by removing layers not in INCLUSION_LIST (which are not connected anymore)
    # Add some additional data like vertices boundaries and layer index
    idx = 0
    cur_lb = 0
    cur_ub = 0
    n_components = 0
    for name in list(layer_graph.keys()):
        node = layer_graph[name]
        if not is_included(node.get_object(model)):
            del layer_graph[name]
        else:
            cur_lb += n_components
            node.idx = idx
            module = node.get_object(model)
            n_components = get_number_components(module)
            cur_ub += n_components
            node.boundaries = [cur_lb, cur_ub - 1]
            idx += 1

    return LayerGraph(model, layer_graph)


def get_number_components(module):
    for property in PROPERTY_NAMES:
        if hasattr(module, property):
            return getattr(module, property)

    raise AttributeError(
        f"Module {module} is not currently supported - file an issue on Github to provide your point of view on how to implement this."
    )


# def change_dim(tns: torch.Tensor, target_dim: int) -> torch.Tensor:
#     """
#     Expand or shrink dimension of tns so that it matches target dim.
#
#     To achieve this: squeeze or unsqueeze first dimension
#     """
#     cur_dim = tns.dim()
#     cur_dim_smaller = cur_dim < target_dim
#     while cur_dim != target_dim:
#         if cur_dim_smaller:
#             tns = tns.unsqueeze(0)
#         else:
#             tns = tns.squeeze(0)
#         print(tns)
#         print(target_dim, cur_dim)
#         input()
#         cur_dim = tns.dim()
#     return tns
#

# def layer_to_vertices(module: nn.Module, node_feature_dim: int = 1) -> torch.Tensor:
#     n_components = get_number_components(module)
#     parameters = list(module.parameters())  # Parameters as list
#     n_items = len(parameters)  # e.g. Linear + bias
#     nodes = [None] * n_components
#     for component in range(n_components):
#         cur_node = [None] * n_items
#         for item in range(n_items):
#             params = parameters[item].data[component]
#             # params = change_dim(
#             #     params, node_feature_dim
#             # )  # TODO remove or change, this doesn't always work...
#             cur_node[item] = params
#         nodes[component] = torch.cat(cur_node)
#     return torch.cat(nodes, dim=0)


def model_to_ajd_list(model: nn.Module) -> tg.data.Data:
    layer_graph = model_to_layer_graph(model)
    print(layer_graph.idx_graph)
    # for idx in range(len(layer_graph)):
    # node_obj = layer_graph.get_node_object(idx)
    # x = layer_to_vertices(node_obj)
    # print(x)
    # for child_names in layer_node.children:
    #     # TODO Create vertices
