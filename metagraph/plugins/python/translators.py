from metagraph import translator
from metagraph.plugins import has_grblas
from .types import PythonNodeMapType, PythonNodeSetType
from ..numpy.types import NumpyNodeMap, NumpyNodeSet


@translator
def nodemap_to_nodeset(x: PythonNodeMapType, **props) -> PythonNodeSetType:
    return set(x)


@translator
def nodeset_from_numpy(x: NumpyNodeSet, **props) -> PythonNodeSetType:
    return set(x.value.tolist())


@translator
def nodemap_from_numpy(x: NumpyNodeMap, **props) -> PythonNodeMapType:
    return dict(zip(x.nodes.tolist(), x.value.tolist()))


@translator
def nodeset_from_numpy_nodemap(x: NumpyNodeMap, **props) -> PythonNodeSetType:
    return set(x.nodes.tolist())


if has_grblas:
    from ..graphblas.types import GrblasNodeMap

    @translator
    def nodemap_from_graphblas(x: GrblasNodeMap, **props) -> PythonNodeMapType:
        idx, vals = x.value.to_values()
        return dict(zip(idx.tolist(), vals.tolist()))
