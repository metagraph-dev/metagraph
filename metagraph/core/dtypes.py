import numpy as np

bool = np.dtype(np.bool)
int8 = np.dtype(np.int8)
int16 = np.dtype(np.int16)
int32 = np.dtype(np.int32)
int64 = np.dtype(np.int64)
uint8 = np.dtype(np.uint8)
uint16 = np.dtype(np.uint16)
uint32 = np.dtype(np.uint32)
uint64 = np.dtype(np.uint64)
float32 = np.dtype(np.float32)
float64 = np.dtype(np.float64)
# Include less common dtypes for increased compatibility with numpy ecosystem
float16 = np.dtype(np.float16)
complex64 = np.dtype(np.complex64)
complex128 = np.dtype(np.complex128)
datetime64 = np.dtype(np.datetime64)
timedelta64 = np.dtype(np.timedelta64)
object = np.dtype(np.object)

_dtypes = [
    bool,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    complex64,
    complex128,
    datetime64,
    timedelta64,
    object,
]
_dtypes = dict(zip(_dtypes, _dtypes))


def dtype(datatype):
    """
    Convert a dtype-like thing to a metagraph data type.

    >>> dtype(float)
    dtype('float64')

    >>> dtype(np.int32)
    dtype('int32')

    >>> dtype('bool')
    dtype('bool')

    """
    typ = np.dtype(datatype)
    if typ not in _dtypes:
        raise ValueError(
            f"dtype not understood by metagraph: {datatype!r} converted to {typ!r}"
        )
    return _dtypes[typ]  # return the exact metagraph dtype


dtypes_simplified = {
    bool: "bool",
    int8: "int",
    int16: "int",
    int32: "int",
    int64: "int",
    uint8: "int",
    uint16: "int",
    uint32: "int",
    uint64: "int",
    float16: "float",
    float32: "float",
    float64: "float",
    complex64: "str",
    complex128: "str",
    datetime64: "str",
    timedelta64: "str",
    object: "str",
}
