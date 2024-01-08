import torch.nn as nn

from .graph import LayerNode

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
