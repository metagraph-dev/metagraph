.. _algorithms:

Algorithms
==========

Algorithms perform the graph computations in Metagraph. They can also be used as helper functions
or anywhere that type conversion needs additional inputs.

For example, creating a NodeMap from a NodeSet by setting each value to 1 would require an algorithm.

Translators can only convert between types without the need for additional inputs to the function.

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
    def strongly_connected_components(graph: EdgeSet(is_directed=True)) -> NodeMap:
        pass

The ``abstract_algorithm`` decorator is required. The full path of the algorithm is passed to
the decorator. This determines how the algorithm will be found below ``resolver.algos``.

The actual name of the Python function does not matter. By convention, it should match the final
name in the algorithm path.

Typing is required in the function signature as this is used to validate concrete versions of
the algorithm. Required properties should be passed in to the abstract type constructor.
If an argument is a plain Python type (like bool or int), that should be indicated as such.
Default values may be listed for plain Python types.

The body of the function is irrelevant. Convention is to simply ``pass``.


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
    def nx_strongly_connected_components(graph: NetworkXEdgeSet) -> PythonNodeMap:
        index_to_label = dict()
        for i, nodes in enumerate(nx.strongly_connected_components(graph.value)):
            for node in nodes:
                index_to_label[node] = i
        return PythonNodeMap(index_to_label)

The ``concrete_algorithm`` decorator is required. The full path of the algorithm must match
the abstract algorithm's path exactly. This is the piece that links the concrete version to
the abstract definition.

As with the abstract algorithm, the actual name of the Python function does not matter.
There is no convention for what it should be named.

Notice that the signature matches the abstract algorithm, but choosing a concrete type rather
than the abstract type.

The body of the function can be as complex as needed. Often, when building a plugin for an
existing library, the concrete algorithm body is quite short because it calls the underlying
library's implementation. The only thing else to do is package the results in the correct
concrete type and return.

*Open issue:* How to add custom parameters that can be passed when calling exact algorithms?


Algorithm Versions
------------------

A planned feature of Metagraph is to version algorithm signatures and paths. This will allow
Metagraph to grow without breaking existing plugins.

Having different versions of the algorithm API will allow plugins to update asynchronously from
core Metagraph. This allows for algorithm signature changes as well as rearranging of the
algorithm hierarchy and path structure as things evolve over time.
