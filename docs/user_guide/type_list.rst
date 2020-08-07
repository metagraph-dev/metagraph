.. _type_list:

List of Core Types
==================

The following are core types in Metagraph. Below each is a description and list of concrete types.
Each concrete type indicates its ``value_type`` and public-facing data objects.


Vector
------

1-D homogeneous array of data

Abstract Properties:

- is_dense: [True, False]
- dtype: ["str", "float", "int", "bool"]

→ Grblas Vector
~~~~~~~~~~~~~~~

:ConcreteType: ``GrblasVectorType``
:value_type: ``grblas.Vector``

→ Numpy Vector
~~~~~~~~~~~~~~

:ConcreteType: ``NumpyVector.Type``
:value_type: ``NumpyVector`` wrapper
:data objects:
    ``.value``: numpy array (1D) of values

    ``.mask``: optional boolean numpy array indicating non-missing values


Matrix
------

2-D homogeneous array of data

Abstract Properties:

- is_dense: [True, False]
- is_square: [True, False]
- dtype: ["str", "float", "int", "bool"]


→ Grblas Matrix
~~~~~~~~~~~~~~~

:ConcreteType: ``GrblasMatrixType``
:value_type: ``grblas.Matrix``

→ Numpy Matrix
~~~~~~~~~~~~~~

:ConcreteType: ``NumpyMatrix.Type``
:value_type: ``NumpyMatrix`` wrapper
:data objects:
    ``.value``: numpy array (2D) of values

    ``.mask``: optional boolean numpy array indicating non-missing values

→ Scipy Matrix
~~~~~~~~~~~~~~

:ConcreteType: ``ScipyMatrixType``
:value_type: ``scipy.sparse.spmatrix``


DataFrame
---------

2-D table of data where each column has a unique name and may have a unique dtype.

Abstract Properties:

- *<none>*

→ Pandas DataFrame
~~~~~~~~~~~~~~~~~~

:ConcreteType: ``PandasDataFrameType``
:value_type: ``pandas.DataFrame``


NodeSet
-------

A set of NodeIDs.

Abstract Properties:

- *<none>*

Standard Wrapper Methods:

- ``num_nodes() -> int``
- ``__contains__(NodeID) -> bool``

→ Grblas NodeSet
~~~~~~~~~~~~~~~~

:ConcreteType: ``GrblasNodeSet.Type``
:value_type: ``GrblasNodeSet``
:data objects:
    ``.value``: grblas.Vector with missing values indicating the NodeID is not part of the set

The ``dtype`` of the Vector is not restricted. The only indication of existence in the set
is that the value is not missing. There is no guarantee of what the value actually is.

→ Numpy NodeSet
~~~~~~~~~~~~~~~

Concrete Properties:

- is_compact: [True, False]

:ConcreteType: ``NumpyNodeSet.Type``
:value_type: ``NumpyNodeSet``
:data objects:
    ``.node_array``: numpy array of all NodeIDs in sorted order

    ``.node_set``: Python set of all NodeIDs

    ``.mask``: boolean numpy array indicating presence of NodeID in set

Either ``mask`` will be set *or* ``node_array`` and ``node_set`` will be set.
The mask will be None when ``is_compact==True``.

To get a numpy array of the nodes, use the ``.nodes()`` method.

→ Python NodeSet
~~~~~~~~~~~~~~~~

:ConcreteType: ``PythonNodeSet.Type``
:value_type: ``PythonNodeSet``
:data objects:
    ``.value``: Python set of NodeIds


NodeMap
-------

A set of NodeIDs and associated values, one for each node.

Abstract Properties:

- dtype: ["str", "float", "int", "bool"]

Can be translated to:

- NodeSet

Standard Wrapper Methods:

- ``num_nodes() -> int``
- ``__contains__(NodeID) -> bool``
- ``__getitem__(NodeID) -> Any``

→ Grblas NodeMap
~~~~~~~~~~~~~~~~

:ConcreteType: ``GrblasNodeMap.Type``
:value_type: ``GrblasNodeMap``
:data objects:
    ``.value``: grblas.Vector containing values for NodeIDs; missing values are not in the set of nodes

→ Numpy NodeMap
~~~~~~~~~~~~~~~

Concrete Properties:

- is_compact: [True, False]

:ConcreteType: ``NumpyNodeMap.Type``
:value_type: ``NumpyNodeMap``
:data objects:
    ``.value``: numpy array of values

    ``.mask``: boolean numpy array indicating presence of NodeIDs in map

    ``.id2pos``: Python dict mapping NodeID to position in ``value``

    ``.pos2id``: numpy array of all NodeIDs in sorted order

For the compact mode, ``mask`` will be None. ``value`` will be dense, corresponding
to NodeIDs in ``pos2id``.

For the non-compact mode, ``id2pos`` and ``pos2id`` will be None. ``value`` will be sparse
with valid data corresponding to True entries in the ``mask``.

→ Python NodeMap
~~~~~~~~~~~~~~~~

:ConcreteType: ``PythonNodeMap.Type``
:value_type: ``PythonNodeMap``
:data objects:
    ``.value``: a Python dict mapping NodeID to value


EdgeSet
-------

A set of edges connecting nodes.

Abstract Properties:

- is_directed: [True, False]

→ Grblas EdgeSet
~~~~~~~~~~~~~~~~

:ConcreteType: ``GrblasEdgeSet.Type``
:value_type: ``GrblasEdgeSet``
:data objects:
    ``.value``: grblas.Matrix representing an adjacency matrix

    ``.transposed``: bool

The indices of the matrix indicate the NodeIDs of the edges.

Missing values in the matrix indicate the edge is not in the set. If there is a value, the edge
is part of the set, but the dtype is not restricted (i.e. don't assume boolean or 1/0).

→ Pandas EdgeSet
~~~~~~~~~~~~~~~~

:ConcreteType: ``PandasEdgeSet.Type``
:value_type: ``PandasEdgeSet``
:data objects:
    ``.value``: pandas.DataFrame with 2 columns

    ``.src_label``: str name of column containing source NodeIDs

    ``.dst_label``: str name of column containing destination NodeIDs

    ``.is_directed``: bool indicating whether to assume directed edges

    ``.index``: pre-built pandas MultiIndex of (src_label, dst_label) tuples

If ``is_directed`` is False, edges are not duplicated in both directions to save space.


→ Scipy EdgeSet
~~~~~~~~~~~~~~~

:ConcreteType: ``ScipyEdgeSet.Type``
:value_type: ``ScipyEdgeSet``
:data objects:
    ``.value``: scipy.sparse matrix representing an adjacency matrix

    ``.node_list``: numpy array of NodeIDs corresponding to indices in the matrix

    ``.transposed``: bool

The indices of the matrix do not represent NodeIDs. Instead, they represent positions within
``node_list`` which holds the actual NodeIDs. If only ``n`` nodes exist in the edge set,
the matrix will be ``n x n``.

There is no guarantee for the matrix dtype. Presence or absence of a value is the only
indication that the edge exists in the edge set.

EdgeMap
-------

A set of edges connecting nodes. Each edge is associated with a value (i.e. weight).

Abstract Properties:

- is_directed: [True, False]
- dtype: ["str", "float", "int", "bool"]
- has_negative_weights: [True, False]

Can be translated to:

- EdgeSet

→ Grblas EdgeMap
~~~~~~~~~~~~~~~~

:ConcreteType: ``GrblasEdgeMap.Type``
:value_type: ``GrblasEdgeMap``
:data objects:
    ``.value``: grblas.Matrix

    ``.transposed``: bool

The indices of the matrix indicate the NodeIDs of the edges.

Values in the matrix are the weighted edges.

→ Pandas EdgeMap
~~~~~~~~~~~~~~~~

:ConcreteType: ``PandasEdgeMap.Type``
:value_type: ``PandasEdgeMap``
:data objects:
    ``.value``: pandas.DataFrame with 3 columns

    ``.src_label``: str name of column containing source NodeIDs

    ``.dst_label``: str name of column containing destination NodeIDs

    ``.weight_label``: str name of column containing the weights

    ``.is_directed``: bool indicating whether to assume directed edges

    ``.index``: pre-built pandas MultiIndex of (src_label, dst_label) tuples

If ``is_directed`` is False, edges are not duplicated in both directions to save space.

→ Scipy EdgeMap
~~~~~~~~~~~~~~~

:ConcreteType: ``ScipyEdgeMap.Type``
:value_type: ``ScipyEdgeMap``
:data objects:
    ``.value``: scipy.sparse matrix representing an adjacency matrix

    ``.node_list``: numpy array of NodeIDs corresponding to indices in the matrix

    ``.transposed``: bool

The indices of the matrix do not represent NodeIDs. Instead, they represent positions within
``node_list`` which holds the actual NodeIDs. If only ``n`` nodes exist in the edge set,
the matrix will be ``n x n``.

The values in the matrix are the edge weights.

The format of the scipy sparse matrix (csr, csc, coo, dok, lil) is not constrained.
Use the ``.format()`` method to check.

*Note about zeros*: scipy sparse assumes missing values are equivalent to zeros.
Few if any other graph libraries make this assumption because it makes it impossible
to differentiate between edges with a weight of 0 and the lack of an edge. Care must
be taken when using the scipy sparse matrix to avoid surprises resulting from this
conflation of ideas.

Graph
-----

A combination of edges and nodes, each of which may hold values or not.
Additionally, a Graph may have orphan nodes (containing no edges), which
an EdgeSet/Map cannot have.

Abstract Properties:

- is_directed: [True, False]
- node_type: ["set", "map"]
- node_dtype: ["str", "float", "int", "bool", None]
- edge_type: ["set", "map"]
- edge_dtype: ["str", "float", "int", "bool", None]
- edge_has_negative_weights: [True, False, None]

Can be translated to:

- NodeSet
- EdgeSet

→ Grblas Graph
~~~~~~~~~~~~~~

:ConcreteType: ``GrblasGraph.Type``
:value_type: ``GrblasGraph``
:data objects:
    ``.edges``: ``GrblasEdgeSet`` or ``GrblasEdgeMap``

    ``.nodes``: optional ``GrblasNodeSet`` or ``GrblasNodeMap``

If ``nodes`` is None, the nodes are assumed to be fully represented by the nodes in the
EdgeSet or EdgeMap.

→ NetworkX Graph
~~~~~~~~~~~~~~~~

:ConcreteType: ``NetworkXGraph.Type``
:value_type: ``NetworkXGraph``
:data objects:
    ``.value``: nx.Graph or nx.DiGraph

    ``.node_weight_label``: key within the node attrs containing the weight

    ``.edge_weight_label``: key within the edge attrs containing the weight

NodeIDs are required to be integers, which is a restriction imposed by Metagraph
to allow for consistent representation by other Graph types. If non-integer
labels are desired, use :ref:`node_labels`.

If any node has a weight, all nodes must have a weight.

If any edge has a weight, all edges must have a weight.

→ Scipy Graph
~~~~~~~~~~~~~

:ConcreteType: ``ScipyGraph.Type``
:value_type: ``ScipyGraph``
:data objects:
    ``.edges``: ``ScipyEdgeSet`` or ``ScipyEdgeMap``

    ``.nodes``: optional ``NumpyNodeSet`` or ``NumpyNodeMap``

If ``nodes`` is None, the nodes are assumed to be fully represented by the nodes in the
EdgeSet or EdgeMap.


BipartiteGraph
--------------

Representation of a bipartite graph with two unique node groups (0 and 1) and
edges which exist only between nodes from different node groups. Like Graphs,
nodes and edges may have values.

Abstract Properties:

- is_directed: [True, False]
- node0_type: ["set", "map"]
- node1_type: ["set", "map"]
- node0_dtype: ["str", "float", "int", "bool", None]
- node1_dtype: ["str", "float", "int", "bool", None]
- edge_type: ["set", "map"]
- edge_dtype: ["str", "float", "int", "bool", None]
- edge_has_negative_weights: [True, False, None]

Can be translated to:

- EdgeSet

→ NetworkX BipartiteGraph
~~~~~~~~~~~~~~~~~~~~~~~~~~

:ConcreteType: ``NetworkXBipartiteGraph.Type``
:value_type: ``NetworkXBipartiteGraph``
:data objects:
    ``.value``: nx.Graph or nx.DiGraph

    ``.nodes``: 2-tuple of sets of NodeIDs

    ``.node_weight_label``: key within the node attrs containing the weight

    ``.edge_weight_label``: key within the edge attrs containing the weight

The two node groups within the bipartite graph are represented by their position
within ``nodes``.

If any node has a weight, all nodes must have a weight. This includes nodes from
both node sets 0 and 1.

If any edge has a weight, all edges must have a weight.
