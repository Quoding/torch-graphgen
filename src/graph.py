from dataclasses import dataclass, field
from operator import attrgetter
from typing import Optional

import torch.nn as nn


@dataclass
class LayerNode:
    name: str
    # TorchFX graph is incoherent - sometimes str, sometimes object ref.
    target: object  # Upstream / towards the input
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

    def get_node(self, node_name):
        return self.graph[node_name]

    def get_node_object(self, node_name):
        node = self.graph[node_name]
        return node.get_object(self.model)
