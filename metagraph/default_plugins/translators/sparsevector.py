from ... import translator
from ..wrappers.pythonobj import PythonSparseVector
from ..wrappers.numpyobj import NumpySparseVector
from ..wrappers.graphblasobj import GrblasVector, dtype_mg_to_grblas
from .. import registry


try:

    @translator(registry=registry)
    def translate_sparsevector_py2np(
        x: PythonSparseVector, **props
    ) -> NumpySparseVector:
        import numpy as np

        data = np.empty((len(x),))
        data[:] = np.nan  # default missing value
        for idx, val in x.obj.items():
            data[idx] = val
        return NumpySparseVector(data, missing_value=np.nan)


except (ImportError, AttributeError):
    pass


try:

    @translator(registry=registry)
    def translate_sparsevector_np2py(
        x: NumpySparseVector, **props
    ) -> PythonSparseVector:
        missing = x.missing_value
        if missing == missing:
            data = {idx: val for idx, val in enumerate(x.obj) if val != missing}
        else:
            # Special case handling for np.nan where nan != nan
            data = {idx: val for idx, val in enumerate(x.obj) if val == val}
        return PythonSparseVector(data, size=len(x))


except (ImportError, AttributeError):
    pass


try:

    @translator(registry=registry)
    def translate_sparsevector_py2grb(x: PythonSparseVector, **props) -> GrblasVector:
        import grblas

        idx, vals = zip(*x.obj.items())
        vec = grblas.Vector.new_from_values(
            idx, vals, size=len(x), dtype=dtype_mg_to_grblas[x.dtype]
        )
        return vec


except (ImportError, AttributeError):
    pass


try:

    @translator(registry=registry)
    def translate_sparsevector_grb2py(x: GrblasVector, **props) -> PythonSparseVector:
        idx, vals = x.to_values()
        data = {k: v for k, v in zip(idx, vals)}
        return PythonSparseVector(data, size=len(x))


except (ImportError, AttributeError):
    pass


try:

    @translator(registry=registry)
    def translate_sparsevector_np2grb(x: NumpySparseVector, **props) -> GrblasVector:
        import grblas

        missing = x.missing_value
        if missing == missing:
            idx, vals = zip(
                *((idx, val) for idx, val in enumerate(x.obj) if val != missing)
            )
        else:
            # Special case handling for np.nan where nan != nan
            idx, vals = zip(
                *((idx, val) for idx, val in enumerate(x.obj) if val == val)
            )
        vec = grblas.Vector.new_from_values(
            idx, vals, size=len(x), dtype=dtype_mg_to_grblas[x.dtype]
        )
        return vec


except (ImportError, AttributeError):
    pass


try:

    @translator(registry=registry)
    def translate_sparsevector_grb2np(x: GrblasVector, **props) -> NumpySparseVector:
        import numpy as np

        inds, vals = x.to_values()
        data = np.empty((x.size,))
        data[:] = np.nan  # default missing value
        for idx, val in zip(inds, vals):
            data[idx] = val
        return NumpySparseVector(data, missing_value=np.nan)


except (ImportError, AttributeError):
    pass
