import torch
import torch.fx
from torch.fx.immutable_collections import immutable_list
import torch.nn as nn
from utils import LayerGraph, LayerNode

INCLUSION_LIST = [
    nn.Linear,
    nn.Conv2d,
    nn.Conv1d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.LazyConv1d,
    nn.LazyConv2d,
    nn.LazyConv3d,
    nn.LazyConvTranspose1d,
    nn.LazyConvTranspose2d,
    nn.LazyConvTranspose3d,
    nn.Bilinear,
    nn.LazyLinear,
    nn.Embedding,
    nn.EmbeddingBag,
    nn.Transformer,
    nn.TransformerEncoder,
    nn.TransformerDecoder,
    nn.TransformerEncoderLayer,
    nn.TransformerDecoderLayer,
]
PROPERTY_NAMES = ["out_channels", "out_features"]


def is_included(module: nn.Module):
    """
    Check if module is included in the INCLUSION_LIST of types

    :param module: module to check membership
    """
    for inclusion in INCLUSION_LIST:
        if isinstance(module, inclusion):
            return True
    return False


def search_for_next_included_layer(
    model: nn.Module,
    node: LayerNode,
    attr: str,
    layer_graph: dict,
    next_clean_nodes_names: list,
) -> list[str]:
    """
    Recursively search the next layer with a type in INCLUSION_LIST

    :param model: neural network model
    :param node: node representing current layer
    :param attr: either 'parents' or 'children', seeking node's attributes
    :param layer_graph: dictionary mapping node names to nodes
    :param next_clean_nodes_names: list of new nodes names that will be the parents or children of `node`, passed by reference
    :return: new list of node names that are either `children` or `parent` of `node` that have layer types in INCLUSION_LIST
    """
    next_nodes: list[LayerNode] = [layer_graph[name] for name in getattr(node, attr)]
    for next_node in next_nodes:
        obj = next_node.get_object(model)
        if not is_included(obj):
            search_for_next_included_layer(
                model, next_node, attr, layer_graph, next_clean_nodes_names
            )
        else:
            next_clean_nodes_names.append(next_node.name)
    return next_clean_nodes_names


def generate_graph_from_model(model):
    model.eval()

    computational_graph = torch.fx.symbolic_trace(model)

    layer_graph = {}

    # Create base nodes, no edges are created
    for node in computational_graph.graph.nodes:
        layer_graph[node.name] = LayerNode(
            node.name,
            node.target,
        )

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

    # Clean up the graph by removing layers not in INCLUSION_LIST
    for name in list(layer_graph.keys()):
        node = layer_graph[name]
        if not is_included(node.get_object(model)):
            del layer_graph[name]

    return LayerGraph(model, layer_graph)
