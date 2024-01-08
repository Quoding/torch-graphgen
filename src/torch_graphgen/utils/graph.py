from dataclasses import dataclass, field
from operator import attrgetter
from typing import Optional, Union

import torch.nn as nn


@dataclass
class LayerNode:
    name: str
    # TorchFX graph is incoherent - sometimes str, sometimes object ref.
    target: object  # Upstream / towards the input
    idx: Optional[int] = None
    parents: Optional[list[str]] = field(default_factory=list)
    # Downstream / towards the output
    children: Optional[list[str]] = field(default_factory=list)

    def get_object(self, model):
        try:
            if type(self.target) == str:
                getter = attrgetter(self.target)
                node_object_ref = getter(model)
            else:
                node_object_ref = self.target
        except:
            node_object_ref = None
        return node_object_ref


@dataclass
class LayerGraph:
    model: nn.Module
    graph: dict
    idx_graph: dict = field(default_factory=dict)

    def __post_init__(self):
        self.create_idx_graph()

    def create_idx_graph(self):
        for node in self.graph.values():
            self.idx_graph[node.idx] = node

    def get_node_by_name(self, node_name):
        return self.graph[node_name]

    def get_node_by_idx(self, idx):
        return self.idx_graph[idx]

    def get_node_object(self, node_identifier: Union[int, str]):
        if isinstance(node_identifier, str):
            node = self.graph[node_identifier]
        else:
            node = self.idx_graph[node_identifier]

        return node.get_object(self.model)

    def __len__(self):
        return len(self.graph)