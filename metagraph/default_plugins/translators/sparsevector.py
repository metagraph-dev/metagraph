from ... import translator
from ..wrappers.python import PythonSparseVector
from ..wrappers.numpy import NumpySparseVector
from ..wrappers.graphblas import GrblasVectorType, dtype_mg_to_grblas
from .. import registry, numpy, grblas


if numpy:
    np = numpy

    @translator(registry=registry)
    def translate_sparsevector_py2np(
        x: PythonSparseVector, **props
    ) -> NumpySparseVector:
        data = np.empty((len(x),))
        data[:] = np.nan  # default missing value
        for idx, val in x.value.items():
            data[idx] = val
        return NumpySparseVector(data, missing_value=np.nan)

    @translator(registry=registry)
    def translate_sparsevector_np2py(
        x: NumpySparseVector, **props
    ) -> PythonSparseVector:
        data = {
            idx: x.value[idx]
            for idx, is_missing in enumerate(x.get_missing_mask())
            if not is_missing
        }
        return PythonSparseVector(data, size=len(x))


if grblas:

    @translator(registry=registry)
    def translate_sparsevector_py2grb(
        x: PythonSparseVector, **props
    ) -> GrblasVectorType:
        import grblas

        idx, vals = zip(*x.value.items())
        vec = grblas.Vector.new_from_values(
            idx, vals, size=len(x), dtype=dtype_mg_to_grblas[x.dtype]
        )
        return vec

    @translator(registry=registry)
    def translate_sparsevector_grb2py(
        x: GrblasVectorType, **props
    ) -> PythonSparseVector:
        idx, vals = x.to_values()
        data = {k: v for k, v in zip(idx, vals)}
        return PythonSparseVector(data, size=x.size)


if numpy and grblas:
    np = numpy

    @translator(registry=registry)
    def translate_sparsevector_np2grb(
        x: NumpySparseVector, **props
    ) -> GrblasVectorType:
        import grblas

        idx = [
            idx for idx, is_missing in enumerate(x.get_missing_mask()) if not is_missing
        ]
        vals = [x.value[i] for i in idx]
        vec = grblas.Vector.new_from_values(
            idx, vals, size=len(x), dtype=dtype_mg_to_grblas[x.dtype]
        )
        return vec

    @translator(registry=registry)
    def translate_sparsevector_grb2np(
        x: GrblasVectorType, **props
    ) -> NumpySparseVector:
        import numpy as np

        inds, vals = x.to_values()
        data = np.empty((x.size,))
        data[:] = np.nan  # default missing value
        for idx, val in zip(inds, vals):
            data[idx] = val
        return NumpySparseVector(data, missing_value=np.nan)
