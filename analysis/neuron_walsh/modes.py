"""Mode registry for the neuron-level Walsh analysis.

The DEFAULT tracked-mode set is exactly the task's binary-tree target modes,
built from parity_net.data.tree_parity_specs (for k=16: 8 d2 + 4 d4 + 2 d8 +
1 d16 = 15 modes), with full-tree enumeration regardless of exclude_targets.
Other Walsh modes may be appended optionally but never replace the tree set.
"""
from __future__ import annotations

import math

from parity_net.data import tree_parity_specs


def build_mode_registry(relevant_dim: int, include_singletons: bool = False) -> list[dict]:
    """[{index, name, support, degree, level, parent, children}] in tree order
    (optionally the degree-1 singletons first, then d2, d4, ..., root).

    The DEFAULT tracked set stays the 15 tree targets; singletons are an
    optional extension (they are not task targets, but corner-detector neurons
    carry 2/3 of their variance in them — see the coverage investigation)."""
    specs = tree_parity_specs(relevant_dim)  # full tree, ignores supervision
    modes: list[dict] = []
    if include_singletons:
        for j in range(relevant_dim):
            modes.append(dict(index=len(modes), name=f"d1_{j}", support=(j,),
                              degree=1, level=0, parent=f"d2_{j // 2}",
                              children=[]))
    for _, s in enumerate(specs):
        deg = s.degree
        i = int(s.name.split("_", 1)[1])
        parent = f"d{deg * 2}_{i // 2}" if deg < relevant_dim else None
        children = [f"d{deg // 2}_{2 * i}", f"d{deg // 2}_{2 * i + 1}"] if deg > 2 else []
        modes.append(dict(
            index=len(modes),
            name=s.name,
            support=tuple(s.indices),
            degree=deg,
            level=int(math.log2(deg)),
            parent=parent,
            children=children,
        ))
    return modes
