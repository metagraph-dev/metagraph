.. _plugin_parts:

Plugin Parts
============

Prerequisite Reading
--------------------

Familiarity with the concepts covered in the following sections are highly recommended:

* :ref:`User Guide<user_guide>`

Introduction
------------

There are three classes of functionality provided by a plugin:

* :ref:`Algorithms<algorithms>` (e.g. abstract algorithms and concrete algorithms)
* :ref:`Types<types>` (e.g. abstract types, concrete types, and wrappers)
* :ref:`Translators<translators>`

This document will cover recommendations to consider when writing plugins as they relate to these three classes of functionality.

It is not necessary for a plugin to provide functionality that fits into all of the above, e.g. a plugin with only translators is valid.

Algorithms
----------

A plugin can provide algorithms.
This can be done in one of two forms, abstract (i.e. a spec) or concrete (i.e. an implementation).

.. _plugin_parts_abstract_algorithm:

Abstract Algorithm
~~~~~~~~~~~~~~~~~~

An abstract algorithm is a spec. Providing this alone can be useful because other developers can provide different
implementations of it. Read more about abstract algorithms :ref:`here<algorithms>`. However, abstract algorithms
cannot be used without at least one implementation, so it's highly recommended to provide at least 1 concrete
implementation when introducing a new abstract algorithm.

Here's some example code showing how to declare an abstract algorithm:

 .. code-block:: python

     from metagraph import abstract_algorithm
     from metagraph.types import Graph

     @abstract_algorithm("centrality.pagerank")
     def pagerank(
         graph: Graph(edge_type="map", edge_dtype={"int", "float"}),
         damping: float = 0.85,
         maxiter: int = 50,
         tolerance: float = 1e-05,
     ) -> NodeMap:
         pass

The ``abstract_algorithm`` decorator denotes that the function ``pagerank`` specifies an abstract algorithm. How the
decorator are used will be explained in more detail in the :ref:`End-to-End Plugin Pathway<end_to_end_plugin_pathway>`.

The string "centrality.pagerank" denotes the name of the abstract algorithm that the function ``pagerank`` specifies.

Since an abstract algorithm is merely a spec, there's no need to specify a body (which is why the body of ``pagerank``
is only ``pass``).

Take note of the type hints. Type hints are checked at plugin registration time to verify that the signatures of
concrete algorithms match the types of the corresponding abstract algorithm.

Items in parentheses following an abstract type are :ref:`properties<properties>` which describe necessary conditions
that the algorithm expects. In the case of ``pagerank``, the input ``Graph`` must have ``edge_type="map"`` which means
there are values associated with the edges (i.e. weights). The ``edge_dtype={"int", "float"}`` property indicates that
the datatype of those weights must be integer or floating point.

Default parameter values are specified in the abstract algorithm and are inherited by all concrete algorithm implementations.

Concrete Algorithm
~~~~~~~~~~~~~~~~~~

A concrete algorithm is the callable implementation of an abstract algorithm.

Read more about concrete algorithms :ref:`here<algorithms>`.

Here's an example concrete algorithm implementation using `NetworkX <https://networkx.github.io/>`_ of `Page Rank <https://en.wikipedia.org/wiki/PageRank>`_.


 .. code-block:: python

     import networkx as nx
     from metagraph import concrete_algorithm

     @concrete_algorithm("centrality.pagerank")
     def nx_pagerank(
         graph: NetworkXGraph, damping: float, maxiter: int, tolerance: float
     ) -> PythonNodeMap:
         pagerank = nx.pagerank(
             graph.value, alpha=damping, max_iter=maxiter, tol=tolerance, weight=None
         )
         return PythonNodeMap(pagerank)

The ``concrete_algorithm`` decorator denotes that the function ``nx_pagerank`` is a concrete algorithm. How the decorator
is used will be explained in more detail in the :ref:`End-to-End Plugin Pathway<end_to_end_plugin_pathway>`.

The string "centrality.pagerank" denotes the name of the concrete algorithm that the function ``nx_pagerank`` specifies.

Here are some details about how the body of ``nx_pagerank`` implements Page Rank:

* ``graph`` is an instance of the concrete type ``NetworkXGraph``, which is intended to wrap a
  `NetworkX <https://networkx.github.io/>`_ graph. The implementation of ``NetworkXGraph`` is such that the ``value``
  attribute is a ``networkx.Graph``.
* The returned value is an instance of the concrete type ``PythonNodeMap``, which is an implementation of the abstract
  return type ``NodeMap``, specified in the :ref:`previous section<plugin_parts_abstract_algorithm>`.

Note that all the concrete types in the signature are concrete implementations of the corresponding abstract types in
the signature of the abstract implementation.

Abstract properties are not repeated in the concrete signature (e.g. ``edge_dtype`` is not specified). The only
properties which would be indicated in a concrete algorithm are concrete properties -- those properties which are
specific to a ``NetworkXGraph``.

Despite the fact that ``nx_pagerank`` has no default values for ``damping``, ``maxiter``, and ``tolerance``, when the
Metagraph resolver calls "centrality.pagerank", the default values from the abstract algorithm are applied as needed
before calling ``nx_pagerank``.

Types
-----

When providing algorithms, it's useful to additionally provide the types that the algorithms use.

Be sure to read the documentation regarding types from the :ref:`User Guide<types>`.

Abstract Types
~~~~~~~~~~~~~~

New abstract algorithms may require new abstract types.

Here's an example of an abstract type declaration:

 .. code-block:: python

    from metagraph import AbstractType
    class EdgeMap(AbstractType):
        properties = {
            "is_directed": [True, False],
            "dtype": DTYPE_CHOICES,
            "has_negative_weights": [True, False],
        }
        unambiguous_subcomponents = {EdgeSet}

As shown above, abstract types are classes.

If new abstract types are introduced, it's highly recommended (but not strictly required) that the plugin provide at
least 1 concrete implementation of that type (i.e. a concrete type).

The introduction of new abstract types in a plugin are rare. If a plugin requires a new abstract type, consider
proposing it as a core abstract type as well since it might be generally useful. Proposals can be made `here <https://github.com/ContinuumIO/metagraph/issues>`_.

For more about abstract types, see :ref:`here<types>`.

Concrete Types
~~~~~~~~~~~~~~

New concrete algorithms may require different data representations of an existing abstract type or a new abstract type
introduced in a plugin.

 .. code-block:: python

    from metagraph import ConcreteType
    import pandas as pd

    class PandasDataFrameType(ConcreteType, abstract=DataFrame):
        value_type = pd.DataFrame

        @classmethod
        def assert_equal(cls, obj1, obj2, aprops1, aprops2, cprops1, cprops2, *, rel_tol=1e-9, abs_tol=0.0):
            digits_precision = round(-math.log(rel_tol, 10))
            pd.testing.assert_frame_equal(
                obj1, obj2, check_like=True, check_less_precise=digits_precision
            )

Though concrete types are implemented as classes, they have no instances in Metagraph. 

They are classes with attributes and class methods used by the Metagraph resolver to find optimal translations paths.

These classes are merely tools used by the Metagraph resolver to determine how to handle the Python data structures
described by the concrete type.

The attribute ``value_type`` is used to associate a Python type with the concrete type. 

It's highly recommended to add an ``assert_equal`` class method for :ref:`testing purposes<testing_algorithms>`.
``assert_equal`` is a class method that takes two instances of the same concrete type and verifies that they represent
the same underlying data. For example, consider a concrete type for edge list style graphs. Two instances of this
concrete type can represent the same graph but might have their edges in a different order. In this case, ``assert_equal``
would not raise any assertion errors. However, if the edge lists represented different graphs, then an assertion error
would be raised.

For more about concrete types, see :ref:`here<types>`.

Wrappers
~~~~~~~~

Since wrappers automatically introduce concrete types, wrappers are also useful to provide in plugins.

 .. code-block:: python

    class NetworkXEdgeMap(EdgeMapWrapper, abstract=EdgeMap):
        def __init__(
            self, nx_graph, weight_label="weight",
        ):
            self.value = nx_graph
            self.weight_label = weight_label
            self._assert_instance(nx_graph, nx.Graph)

        @classmethod
        def assert_equal(cls, obj1, obj2, props1, props2, *, rel_tol=1e-9, abs_tol=0.0):
            ...
            return

It's conventional to have the underlying data stored in the ``value`` attribute.

It's highly recommended to use the inherited ``_assert_instance`` wrapper method to sanity check types. 

It's highly recommended to add an ``assert_equal`` class method as it gets inherited by the automatically created
concrete type and is useful for :ref:`testing purposes<testing_algorithms>`.

For more about wrappers, see :ref:`here<wrappers>`.

Translators
-----------

When a plugin provides new types (which is often necessary when new algorithms are introduced), it's frequently
necessary to provide translators to have the same underlying data operated on by different plugins (see :ref:`here for
the motivation behind translators<concepts_decoupling_storage_from_algorithms>`).

Here's an example translator:

 .. code-block:: python

    from metagraph.plugins.graphblas.types import GrblasNodeMap
    from metagraph.plugins.python.types import PythonNodeMap

    @translator
    def nodemap_from_graphblas(x: GrblasNodeMap, **props) -> PythonNodeMap:
        idx, vals = x.value.to_values()
        data = dict(zip(idx, vals))
        return PythonNodeMap(data)

The implementation of translators should be as complicated as required to adequately convert from any version of
one type into another type. In many cases, however, the logic can be very simple, as is the case in this example.

The ``translator`` decorator allows the Metagraph resolver to use this translator. How the decorator are used will be
explained in more detail in the :ref:`End-to-End Plugin Pathway<end_to_end_plugin_pathway>`.

Since plugins are more useful when interoperating with other plugins rather than being used in isolation, it's useful
to provide translators that translate to and from concrete types introduced in a new plugin with the rest of the
Metagraph plugin ecosystem.

When writing translators, it's infeasible to write a translator from a single concrete type to every other concrete
type due to the explosive number of possible translation paths. Thus, it's recommended to at least (when possible) write
translators to the core Metagraph concrete types. The core concrete types can act as a translation hub to the
concrete types introduced in external plugins.

For more about translators, see :ref:`here<translators>`.
