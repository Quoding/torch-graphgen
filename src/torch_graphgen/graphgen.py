import os
from typing import Union, List

import torch
from torch.fx.immutable_collections import immutable_list
import torch.nn as nn

from .utils.graph import LayerNode, recording_hook
from .utils.inclusion import is_included, search_for_next_included_layer
from .utils.stats import get_n_components


class LayerGraph:
    def __init__(self, model: nn.Module):
        self.model: nn.Module = model
        self.graph: dict = {}
        self.idx_graph: dict = {}
        self.initial_nodes: List[LayerNode] = []
        # self.adj_list = []
        # self.edge_list = []

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

    def to_component_edge_list(
        self, output: str, parents=True, children=True, overwrite=False
    ):
        assert parents or children
        # assert not os.path.isfile(output)
        if os.path.isfile(output) and overwrite == False:
            print("Output file already exists, skipping writing edge list")
            return
        else:
            print(f"Deleted file {output}. Will proceed with overwrite.")
            os.remove(output)
        edge_list = []

        for cur_node in self.graph.values():
            for cur_vertex in range(*cur_node.boundaries):
                next_nodes = []
                if parents:
                    next_nodes.extend(cur_node.parents)
                if children:
                    next_nodes.extend(cur_node.children)
                for next_node in next_nodes:
                    edges = map(lambda x: [cur_vertex, x], range(*next_node.boundaries))
                    edge_list.extend(edges)

            # Dump current edge_list
            if len(edge_list) > 100_000:
                with open(output, "a") as f:
                    for edge in edge_list:
                        string = " ".join(map(str, edge))
                        f.write(string + "\n")

                edge_list = []
        # Final write
        with open(output, "a") as f:
            for edge in edge_list:
                string = " ".join(map(str, edge))
                f.write(string + "\n")

    def to_component_adj_list(self, output: str, parents=True, children=True):
        assert parents or children
        assert not os.path.isfile(output)

        adj_list = []
        for cur_node in self.graph.values():
            for _ in range(*cur_node.boundaries):
                vertex_adj_list = []
                next_nodes = []
                if parents:
                    next_nodes.extend(cur_node.parents)
                if children:
                    next_nodes.extend(cur_node.children)
                for next_node in next_nodes:
                    vertex_adj_list.extend(list(range(*next_node.boundaries)))
                adj_list.append(vertex_adj_list)

            # Dump current adj_list
            if len(adj_list) > 10_000:
                with open(output, "a") as f:
                    for edge in adj_list:
                        string = " ".join(map(str, edge))
                        f.write(string + "\n")

                adj_list = []
        # Final write
        with open(output, "a") as f:
            for edge in adj_list:
                string = " ".join(map(str, edge))
                f.write(string + "\n")

    # def write_activation_features(self, data: torch.Tensor, output: str):
    #     record_buffer = []
    #     hook = recording_hook(record_buffer)
    #     for cur_node in self.graph.values():
    #         module = self.get_node_module(cur_node.name)
    #         module.register_forward_hook(hook)
    #
    #     self.model(data)
    #     print(cur_node)
    #     # print(record_buffer[-3])
    #     print([record_buffer[i].shape for i in range(len(record_buffer))])
    #     # TODO sortir les features par neurones
    #     # print(len(record_buffer))
    #     # print(self.__len__())

    def __len__(self) -> int:
        return len(self.graph)
