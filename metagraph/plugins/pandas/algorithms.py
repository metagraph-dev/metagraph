from metagraph.plugins import has_pandas
from metagraph import concrete_algorithm
from .types import PandasEdgeSet, PandasEdgeMap
from typing import Any
import numpy as np

if has_pandas:
    import pandas as pd

    @concrete_algorithm("util.edgemap.from_edgeset")
    def pd_edgemap_from_edgeset(
        edgeset: PandasEdgeSet, default_value: Any,
    ) -> PandasEdgeMap:
        new_df = edgeset.value.copy()
        new_df["weight"] = pd.Series(np.full(len(new_df), default_value))
        return PandasEdgeMap(
            new_df,
            src_label=edgeset.src_label,
            dst_label=edgeset.dst_label,
            weight_label="weight",
            is_directed=edgeset.is_directed,
        )
