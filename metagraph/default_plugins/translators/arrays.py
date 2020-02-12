from ... import PluginRegistry, translator
from ..concrete_types.pythonobj import PythonSparseVectorType, PythonSparseVector
from ..concrete_types.numpyobj import NumpySparseVectorType, NumpySparseVector
from ..concrete_types.graphblasobj import GrblasVectorType, dtype_mg_to_grblas

reg = PluginRegistry("metagraph_core")


try:

    @reg.register
    @translator
    def translate_sparsevector_py2np(
        x: PythonSparseVectorType, **props
    ) -> NumpySparseVectorType:
        import numpy as np

        data = np.empty((len(x),))
        data[:] = np.nan  # default missing value
        for idx, val in x.obj.items():
            data[idx] = val
        return NumpySparseVector(data, missing_value=np.nan)


except (ImportError, AttributeError):
    pass


try:

    @reg.register
    @translator
    def translate_sparsevector_np2py(
        x: NumpySparseVectorType, **props
    ) -> PythonSparseVectorType:
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

    @reg.register
    @translator
    def translate_sparsevector_py2grb(
        x: PythonSparseVectorType, **props
    ) -> GrblasVectorType:
        import grblas

        idx, vals = zip(*x.obj.items())
        vec = grblas.Vector.new_from_values(
            idx, vals, size=len(x), dtype=dtype_mg_to_grblas[x.dtype]
        )
        return vec


except (ImportError, AttributeError):
    pass


try:

    @reg.register
    @translator
    def translate_sparsevector_grb2py(
        x: GrblasVectorType, **props
    ) -> PythonSparseVectorType:
        idx, vals = x.to_values()
        data = {k: v for k, v in zip(idx, vals)}
        return PythonSparseVector(data, size=len(x))


except (ImportError, AttributeError):
    pass


try:

    @reg.register
    @translator
    def translate_sparsevector_np2grb(
        x: NumpySparseVectorType, **props
    ) -> GrblasVectorType:
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

    @reg.register
    @translator
    def translate_sparsevector_grb2np(
        x: GrblasVectorType, **props
    ) -> NumpySparseVectorType:
        import numpy as np

        inds, vals = x.to_values()
        data = np.empty((x.size,))
        data[:] = np.nan  # default missing value
        for idx, val in zip(inds, vals):
            data[idx] = val
        return NumpySparseVector(data, missing_value=np.nan)


except (ImportError, AttributeError):
    pass
