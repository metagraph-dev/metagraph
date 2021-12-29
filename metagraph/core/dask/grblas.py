import dask
import distributed
import dask_grblas as dgb
import grblas as gb
import numpy as np
from typing import List


class DaskGrblasLoader:
    @staticmethod
    def register_dask_scheduler_plugin(client: distributed.Client):
        # nothing to do, since Dask will track the lifetime automatically
        pass

    @staticmethod
    def allocate(shape, nvalues, pointers_dtype, indices_dtype, values_dtype):
        # nothing to allocate here.  Going to save this metadata for use in
        # finalize().
        return dict(
            shape=shape,
            nvalues=nvalues,
            pointers_dtype=pointers_dtype,
            indices_dtype=indices_dtype,
            values_dtype=values_dtype,
        )

    @staticmethod
    def dask_incref(csr):
        # nothing to do here, since Dask will track lifetime automatically
        pass

    @staticmethod
    def load_chunk(
        csr,
        row_offset: int,
        pointers: np.ndarray,
        value_offset: int,
        indices: np.ndarray,
        values: np.ndarray,
    ):
        # construct new, self-contained pointers from this slice of the full pointer array
        # if index_value_offset == 0, we know this is the first chunk and should
        # pad out the initial rows
        row_pad = row_offset if value_offset == 0 else 1
        new_pointers = np.zeros(pointers.shape[0] + row_pad, dtype=pointers.dtype)
        new_pointers[row_pad:] = pointers - value_offset  # each chunk is self-contained

        return gb.Matrix.ss.import_csr(
            nrows=new_pointers.shape[0] - 1,
            ncols=csr["shape"][1],
            indptr=new_pointers,
            values=values,
            col_indices=indices,
            sorted_cols=True,
            take_ownership=False,
        )

    @classmethod
    def finalize(cls, csr, plan, chunks: List):
        matrices = [
            dgb.Matrix.from_delayed(
                chunks[i],
                gb.dtypes.lookup_dtype(plan.value_dtype),
                plan.chunks[i].fill_row_end - plan.chunks[i].fill_row_begin + 1,
                plan.matrix_shape[1],
            )
            for i in range(len(chunks))
        ]

        return dgb.row_stack(matrices)
