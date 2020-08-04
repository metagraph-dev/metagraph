from metagraph.plugins import has_pandas
from metagraph import concrete_algorithm
from .types import PandasEdgeSet, PandasEdgeMap
from typing import Any
import numpy as np

if has_pandas:
    import pandas as pd

    @concrete_algorithm("util.edge_map.from_edge_set")
    def pd_edge_map_from_edge_set(
        edge_set: PandasEdgeSet, default_value: Any,
    ) -> PandasEdgeMap:
        new_df = edge_set.value.copy()
        new_df["weight"] = pd.Series(np.full(len(new_df), default_value))
        return PandasEdgeMap(
            new_df,
            src_label=edge_set.src_label,
            dst_label=edge_set.dst_label,
            weight_label="weight",
            is_directed=edge_set.is_directed,
        )
