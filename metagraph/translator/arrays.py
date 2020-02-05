from .base import register_translator
from .. import dataobj


try:

    @register_translator(dataobj.PythonSparseArray, dataobj.NumpyDenseArray)
    def convert(x, **props):
        import numpy as np

        data = np.empty((len(x),))
        data[:] = np.nan  # default missing value
        for idx, val in x.obj.items():
            data[idx] = val
        return dataobj.NumpyDenseArray(data, missing_value=np.nan)


except (ImportError, AttributeError):
    pass


try:

    @register_translator(dataobj.NumpyDenseArray, dataobj.PythonSparseArray)
    def convert(x, **props):
        missing = x.missing_value
        if missing == missing:
            data = {idx: val for idx, val in enumerate(x.obj) if val != missing}
        else:
            # Special case handling for np.nan where nan != nan
            data = {idx: val for idx, val in enumerate(x.obj) if val == val}
        return dataobj.PythonSparseArray(data, size=len(x))


except (ImportError, AttributeError):
    pass


try:

    @register_translator(dataobj.PythonSparseArray, dataobj.GrblasVector)
    def convert(x, **props):
        import grblas

        idx, vals = zip(*x.obj.items())
        vec = grblas.Vector.new_from_values(
            idx, vals, size=len(x), dtype=dataobj.dtype_mg_to_grblas[x.dtype]
        )
        return dataobj.GrblasVector(vec)


except (ImportError, AttributeError):
    pass


try:

    @register_translator(dataobj.GrblasVector, dataobj.PythonSparseArray)
    def convert(x, **props):
        idx, vals = x.obj.to_values()
        data = {k: v for k, v in zip(idx, vals)}
        return dataobj.PythonSparseArray(data, size=len(x))


except (ImportError, AttributeError):
    pass


try:

    @register_translator(dataobj.NumpyDenseArray, dataobj.GrblasVector)
    def convert(x, **props):
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
            idx, vals, size=len(x), dtype=dataobj.dtype_mg_to_grblas[x.dtype]
        )
        return dataobj.GrblasVector(vec)


except (ImportError, AttributeError):
    pass


try:

    @register_translator(dataobj.GrblasVector, dataobj.NumpyDenseArray)
    def convert(x, **props):
        import numpy as np

        inds, vals = x.obj.to_values()
        data = np.empty((len(x),))
        data[:] = np.nan  # default missing value
        for idx, val in zip(inds, vals):
            data[idx] = val
        return dataobj.NumpyDenseArray(data, missing_value=np.nan)


except (ImportError, AttributeError):
    pass
