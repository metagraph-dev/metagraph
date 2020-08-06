.. _type_list:

List of Core Types
==================

The following are core types in Metagraph. Below each is a description and list of concrete types.
Each concrete type indicates its ``value_type``.


Vector
------

1-D homogeneous array of data

- GrblasVectorType -> grblas.Vector
- NumpyVectorType -> NumpyVector wrapper

Matrix
------

2-D homogeneous array of data

- GrblasMatrixType -> grblas.Matrix
- NumpyMatrixType -> NumpyMatrix wrapper
- ScipyMatrixType -> scipy.sparse.spmatrix

DataFrame
---------

2-D table of data where each column has a unique name and may have a unique dtype.

- PandasDataFrameType -> pandas.DataFrame

NodeSet
-------

A set of nodes.

- GrblasNodeSetType -> GrblasNodeSet wrapper
- NumpyNodeSetType -> NumpyNodeSet wrapper
- PythonNodeSetType -> PythonNodeSet wrapper

NodeMap
-------

A set of nodes, with each node containing an associated value.

- GrlbasNodeMapType -> GrlbasNodeMap wrapper
- NumpyNodeMapType -> NumpyNodeMap wrapper
- PythonNodeMapType -> PythonNodeMap wrapper

EdgeSet
-------

A set of edges connecting nodes.

- GrblasEdgeSetType -> GrblasEdgeSet wrapper
- PandasEdgeSetType -> PandasEdgeSet wrapper
- ScipyEdgeSetType -> ScipyEdgeSet wrapper

EdgeMap
-------

A set of edges connecting nodes. Each edge is associated with a value (i.e. weight).

- GrblasEdgeMapType -> GrblasEdgeMap wrapper
- PandasEdgeMapType -> PandasEdgeMap wrapper
- ScipyEdgeMapType -> ScipyEdgeMap wrapper

Graph
-----

A combination of edges and nodes, each of which may hold values or not.

- GrblasGraphType -> GrblasGraph wrapper
- NetworkXGraphType -> NetworkXGraph wrapper
- ScipyGraphType -> ScipyGraph wrapper

BipartiteGraph
--------------

Representation of a bipartite graph with two unique node groups and edges which
exist only between nodes from different node groups. Like Graphs, nodes and
edges may have values.

- NetworkXBipartiteGraphType -> NetworkXBipartiteGraph
