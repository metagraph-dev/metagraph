.. _algorithms:

Algorithms
==========

Algorithms perform the graph computations in Metagraph. They can also be used as helper functions
or anywhere that type conversion needs additional inputs.


Abstract Algorithms
-------------------

Abstract algorithms define the API that the user will call. It indicates which abstract types are
used as inputs and what abstract type will be the output.

Abstract algorithms can also specify required abstract properties that must be satisfied in order for
the algorithm to work properly. For example, strongly connected components needs a directed graph.
Passing in an undirected graph is invalid for the algorithm, regardless of which specific implementation
is chosen.

Abstract algorithm definitions look like this.

.. code-block:: python

    @abstract_algorithm("clustering.strongly_connected_components")
    def strongly_connected_components(graph: Graph(is_directed=True)) -> NodeMap:
        pass

The ``abstract_algorithm`` decorator is required. The full path of the algorithm is passed to
the decorator. This determines how the algorithm will be found below ``resolver.algos``.

The actual name of the Python function does not matter. By convention, it should match the final
name in the algorithm path.

Typing is required in the function signature as this is used to validate concrete versions of
the algorithm. Required properties should be passed in to the abstract type constructor.
If an argument is a plain Python type (like bool or int), that should be indicated as such.
Default values may be listed for plain Python types.

The body of the function is irrelevant for abstract algorithms. Convention is to simply ``pass``.


Concrete Algorithms
-------------------

Concrete algorithms do the actual work on real data objects. They must match the signature of
their corresponding abstract algorithm, but be typed with concrete types rather than abstract
types.

Abstract properties should not be listed in the function signature, but concrete properties
can be indicated by passing them to the concrete type. Default values must never be listed. Instead,
they are inherited from the abstract signature.

Concrete algorithm definitions look like this.

.. code-block:: python

    @concrete_algorithm("clustering.strongly_connected_components")
    def nx_strongly_connected_components(graph: NetworkXGraph) -> PythonNodeMapType:
        index_to_label = dict()
        for i, nodes in enumerate(nx.strongly_connected_components(graph.value)):
            for node in nodes:
                index_to_label[node] = i
        return index_to_label

The ``concrete_algorithm`` decorator is required. The full path of the algorithm must match
the abstract algorithm's path exactly. This is the piece that links the concrete version to
the abstract definition.

As with the abstract algorithm, the actual name of the Python function does not matter.
There is no convention for what it should be named.

Notice that the signature matches the abstract algorithm, but choosing a concrete type rather
than the abstract type. Concrete algorithms must contain a matching entry for every parameter
in the abstract signature, but may contain additional parameters which are specialized to this
implementation.

Each additional parameter must contain a default value. When calling using the normal dispatch
mechanism, these default values will be used. The only way for a user to indicate a value for
the additional parameters is to make an :ref:`exact algorithm call<exact_algorithm_call>`.

The body of the function can be as complex as needed. Often, when building a plugin for an
existing library, the concrete algorithm body is quite short because it calls the underlying
library's implementation. The only thing else to do is package the results in the correct
concrete type and return.


Using the Resolver in Concrete Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a concrete algorithm needs access to the ``Resolver`` object which called it,
set the ``include_resolver`` flag in the decorator and include a ``resolver`` keyword argument
in the signature.

.. code-block:: python

    @concrete_algorithm("path.to.algorithm", include_resolver=True)
    def my_conc_algo(x: NumpyNodeMap, *, resolver) -> int:
        # resolver is now available


Union, List, and Optional types
-------------------------------

Typically, an algorithm parameter has a single type -- either a Metagraph defined type like
``NodeMap`` or a Python type like ``int``.

There are cases, however, where a single type is not sufficient. ``graph_build`` is a good example
as shown below:

.. code-block:: python

    @abstract_algorithm("util.graph.build")
    def graph_build(
        edges: mg.Union[EdgeSet, EdgeMap],
        nodes: mg.Optional[mg.Union[NodeSet, NodeMap]] = None,
    ) -> Graph:
        pass

``edges`` can be either an ``EdgeSet`` or an ``EdgeMap``. ``nodes`` can be one of two possible
types, but can additionally be unspecified (i.e. optional).

To indicate these, we use the standard Python ``typing`` objects ``Union`` and ``Optional``. However,
these are limited to class objects only. In Metagraph, we often need to specialize our types --
``EdgeSet(is_directed=True)`` rather than just ``EdgeSet``. For the specialized case, the regular
Python ``typing.Union`` would fail. To work around this limitation, Metagraph has ``mg.Union``, ``mg.List``, and
``mg.Optional`` which behave identically to the ``typing`` counterparts, but accept classes and instances.
It is recommended to always use the Metagraph versions of ``Union``, ``List``, and ``Optional`` when
defining algorithms in Metagraph.

Interaction between Union and unambiguous_subcomponents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a single type is declared, or is declared as Optional, Metagraph will attempt to translate
input to be compatible with an algorithm signature. With ``unambiguous_subcomponents`` allowing
translation across abstract types, this leads to a nice outcome where passing a ``NodeMap`` to an
algorithm expecting a ``NodeSet`` will just work. The algorithm obviously only needs the set of nodes,
so dropping the weights from the ``NodeMap`` allows the algorithm to still run correctly.

For the case of ``Union``, however, allowing translation across abstract types is problematic.
For the case of ``graph_build``, if we allowed an ``EdgeMap`` to be translated into an ``EdgeSet``,
we would lose critical information. A ``Union`` indicates either is acceptable, but does not indicate
that both are equivalent.

For this reason, when a ``Union`` is used in an algorithm signature, ``unambiguous_subcomponents``
will be ignored for the purpose of translating input objects.


Algorithm Versions
------------------

Metagraph allows algorithms to be versioned. By default, all algorithm signatures define version 0
of the algorithm. To indicate other versions, include the version in the decorator.

.. code-block:: python

    @abstract_algorithm("clustering.strongly_connected_components", version=2)
    def strongly_connected_components(graph: Graph(is_directed=True)) -> NodeMap:
        pass

The algorithm version must be an integer (i.e. no semantic versioning) and should increment one
higher than the previous version.

Algorithms might need to bump their version when the algorithm signature changes, but also to
allow rearranging of the algorithm hierarchy and path structure.

Multiple versions of an algorithm are allowed to be defined within a single release of Metagraph
or a Metagraph plugin. Even though multiple versions are defined, Metagraph will only use the latest
abstract version defined. This keeps the usage of Metagraph simple while allowing plugin authors to
write implementations for multiple releases of Metagraph. This allows plugins to update asynchronously
from core Metagraph.
