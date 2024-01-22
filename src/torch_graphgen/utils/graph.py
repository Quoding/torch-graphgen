from dataclasses import dataclass, field
from operator import attrgetter
from typing import Optional, Union, List

import torch.nn as nn


@dataclass
class LayerNode:
    name: str
    # TorchFX graph is incoherent - sometimes str, sometimes object ref.
    target: object  # Upstream / towards the input
    idx: Optional[int] = None
    boundaries: List[int] = field(default_factory=list)  # [inclusive, exclusive] bounds
    parents: List["LayerNode"] = field(default_factory=list)
    # Downstream / towards the output
    children: List["LayerNode"] = field(default_factory=list)

    def get_module(self, model) -> Union[nn.Module, None]:
        node_object_ref = None
        try:
            if type(self.target) == str:
                getter = attrgetter(self.target)
                node_object_ref = getter(model)
            else:
                node_object_ref = self.target
        except:
            pass

        return node_object_ref

    def __repr__(self):
        return f"{self.name, self.idx, self.boundaries}"


def recording_hook(record_buffer):
    def hook(module, in_, output):
        record_buffer.append(output)

    return hook
