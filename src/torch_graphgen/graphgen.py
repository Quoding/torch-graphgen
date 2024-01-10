from typing import Union
from .utils.inclusion import search_for_next_included_layer, is_included
from .utils.stats import get_n_components
from .utils.graph import LayerNode

import torch.nn as nn
import torch
from torch.fx.immutable_collections import immutable_list


class LayerGraph:
    def __init__(self, model: nn.Module):
        self.model: nn.Module = model
        self.graph: dict = {}
        self.idx_graph: dict = {}
        self.initial_nodes: list[LayerNode] = []
        self.adj_list = []

        model.eval()

        self.gen_layer_graph()
        self.find_initial_nodes()
        self.gen_idx_graph()

    def find_initial_nodes(self):
        self.initial_nodes = [
            node for node in self.graph.values() if len(node.parents) == 0
        ]

    def gen_idx_graph(self):
        if self.idx_graph:
            print(
                "Graph was already generated previously. Call reset() to reset object and regenerate the graph."
            )
            return
        for node in self.graph.values():
            self.idx_graph[node.idx] = node

    def get_node(self, node_identifier: Union[int, str]) -> LayerNode:
        if isinstance(node_identifier, str):
            node = self.graph[node_identifier]
        else:
            node = self.idx_graph[node_identifier]
        return node

    def get_node_module(
        self, node_identifier: Union[int, str]
    ) -> Union[nn.Module, None]:
        node = self.get_node(node_identifier)
        return node.get_module(self.model)

    def reset(self):
        self.__init__(self.model)

    def gen_layer_graph(self):
        if self.graph:
            print(self.graph)
            print(
                "Graph was already generated previously. Call reset() to reset object and regenerate the graph."
            )
            return
        computational_graph = torch.fx.symbolic_trace(self.model)

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
                    parent_node.children.append(cur_node)
                    cur_node.parents.append(parent_node)
            else:
                parent_name = node.args[0].name
                parent_node = layer_graph[parent_name]
                parent_node.children.append(cur_node)
                cur_node.parents.append(parent_node)

        # Connect is_included layers together, clean up unimportant layers
        for node in reversed(computational_graph.graph.nodes):
            if node.op != "placeholder" and node.op != "output":
                cur_node = layer_graph[node.name]

                # Update parents to nearest included layer
                next_clean_nodes = []
                cur_node.parents = search_for_next_included_layer(
                    self.model, cur_node, "parents", layer_graph, next_clean_nodes
                )

                # Update children to nearest included layer
                next_clean_nodes = []
                cur_node.children = search_for_next_included_layer(
                    self.model, cur_node, "children", layer_graph, next_clean_nodes
                )
        # Clean up the graph by removing layers not in INCLUSION_LIST (which are not connected anymore)
        # Add some additional data like vertices boundaries and layer index
        idx = 0
        cur_lb = 0
        cur_ub = 0
        n_components = 0
        for name in list(layer_graph.keys()):
            node = layer_graph[name]
            if not is_included(node.get_module(self.model)):
                del layer_graph[name]
            else:
                cur_lb += n_components
                node.idx = idx
                module = node.get_module(self.model)
                n_components = get_n_components(module)
                cur_ub += n_components
                node.boundaries = [cur_lb, cur_ub]
                idx += 1

        self.graph = layer_graph

    def gen_component_adj_list(self):
        if self.adj_list:
            print(
                "Adjacency list was already generated. Call reset() to regenerate graph, then call this function again."
            )
            return
        global_adj_list = []
        for cur_node in self.graph.values():
            for _ in range(*cur_node.boundaries):
                vertex_adj_list = []
                for child_node in cur_node.children:
                    vertex_adj_list.extend(list(range(*child_node.boundaries)))
                global_adj_list.append(vertex_adj_list)

        self.adj_list = global_adj_list

    def __len__(self) -> int:
        return len(self.graph)
