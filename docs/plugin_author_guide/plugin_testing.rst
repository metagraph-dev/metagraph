.. _plugin_testing:

Plugin Testing
==============

Prerequisite Reading
--------------------

Familiarity with the concepts covered in the following sections are highly recommended:

* :ref:`User Guide<user_guide>`
* :ref:`Plugin Parts<plugin_parts>`
* :ref:`End-to-End Plugin Pathway<end_to_end_plugin_pathway>`

Introduction
------------

This document provides some recommendations when it comes to testing plugins.

This is not a comprehensive guide to testing in general.

We'll cover ways to test the different :ref:`plugin parts<plugin_parts>`.

Testing Abstract Types, Concrete Types, Wrappers
------------------------------------------------

Testing types is difficult to do in isolation outside of trivially testing methods and attributes.

There are no Metagraph-specific testing practices we recommend in particular here.

Much of the functionality provided by types will be rigorously tested indirectly in the algorithm tests and translator tests.

Testing Translators
-------------------

It's highly recommended to write tests for every translator included in a module.

The ``assert_equal`` method of concrete types discussed in :ref:`Plugin Parts<plugin_parts>` comes in handy for translator tests.

Translator tests will frequently come in the form of creating an instance for a concrete type, translating it via the
Metagraph resolver, and then verifying that the translated data structure is as we expected.

Here's an example:

.. code-block:: python

    import metagraph as mg
    from metagraph.plugins.python.types import PythonNodeMap
    from metagraph.plugins.numpy.types import NumpyNodeMap

    def test_python_2_compactnumpy():
        r = mg.resolver
        x = PythonNodeMap({0: 12.5, 1: 33.4, 42: -1.2})
        assert x.num_nodes == 3

        # Convert python -> numpy
        intermediate = NumpyNodeMap(np.array([12.5, 33.4, -1.2]), node_ids=np.array([0, 1, 42]))

        y = r.translate(x, NumpyNodeMap)
        r.assert_equal(y, intermediate)

        # Convert python <- numpy
        x2 = r.translate(y, PythonNodeMap)
        r.assert_equal(x, x2)

Here we test translation from a Python node map to a `NumPy <https://numpy.org/>`_ node map and back again.

We use the Metagraph resolver's ``translate`` method to translate as necessary and ``assert_equal`` method to verify that
the translations are valid. The Metagraph resolver's ``assert_equal`` method utilizes the ``assert_equal`` implemented by
the relevant concrete types.

.. _testing_algorithms:

Testing Algorithms
------------------

Since abstract algorithms are merely specifications, we can only test concrete algorithms.

When testing concrete algorithms, simply testing that outputs for given inputs match expectations is insufficient
for verifying that the outputs also match the results from concrete algorithms written in other plugins (that correspond
to the same abstract algorithm).

We highly recommend using the utility ``metagraph.core.multiverify.MultiVerify`` as it verifies that all concrete
algorithms for a given abstract algorithm give the same result.

It does this by finding all the concrete algorithms for the given abstract algorithm,
using the resolver's translators to translate the input types to the appropriate types for each concrete
algorithm, and using the ``assert_equal`` method of the concrete types to verify that results from all the concrete
algorithms are the same.

This also indirectly tests translators.

MultiVerify and MultiResults
----------------------------

``MultiVerify(r)`` creates a MultiVerify instance bound to the Resolver ``r``.

Following creation, ``.compute()`` is called to exercise all implementations of an abstract algorithm.
The signature for ``compute`` is:

\
    .. code-block:: python

        r = mg.resolver

        mv = MultiVerify(r)
        results = mv.compute("path.to.abstract.algorithm", *args, **kwargs)

The abstract algorithm path is identical to the path used to call the algorithm from the resolver.
For example, "centrality.pagerank" and ``resolver.algos.centrality.pagerank`` are equivalent ways to
point to PageRank abstract algorithm.

The ``args`` and ``kwargs`` of the algorithm are passed in following the algorithm name in a manner similar
to ``functools.partial``.

The result of calling ``compute`` is a ``MultiResult`` instance.

MultiResult
~~~~~~~~~~~

The ``MultiResult`` contains a map of all concrete algorithms and their associated return values.

MultiResults can be used in several ways:

``.normalize(type)``
    Converts all results to a consistent type

``.assert_equal(expected_value)``
    Compares all results against an expected value

``custom_compare(cmp_func)``
    Passes each result to a comparison function

``MultiVerify.transform(exact_algo, *args, **kwargs)``
    The MultiResult is an argument to ``transform``, allowing further refinement of the result using additional
    Metagraph algorithms

``[index]``
    If the concrete algorithms return tuples, slicing the MultiResult will return a new MultiResult
    with each tuple sliced accordingly

normalize
~~~~~~~~~

Calling ``results.normalize(type)`` will perform the translation of those results to the indicated
type. This is a pre-requisite step for several other operations, although it is not required for ``assert_equal``
as the type of the comparison is known and ``normalize`` is called under the hood.

The result of calling ``normalize`` is a new ``MultiResult`` with the same concrete algorithm list, but all results
are of a uniform type.

assert_equal
~~~~~~~~~~~~

``assert_equal`` compares and expected result with each value of the ``MultiResult``. The output of each
algorithm is translated to the same type as the expected result before calling the type's ``assert_equal``
method.

Here's an example:

.. code-block:: python

    import networkx as nx
    import numpy as np
    import metagraph as mg
    from metagraph.core.multiverify import MultiVerify

    r = mg.resolver
    networkx_graph_data = [(0, 1), (0, 2), (2, 0), (1, 2), (3, 2)]
    networkx_graph = nx.DiGraph()
    networkx_graph.add_edges_from(networkx_graph_data)
    data = {
        0: 0.37252685132844066,
        1: 0.19582391181458728,
        2: 0.3941492368569718,
        3: 0.037500000000000006,
    }
    expected_val = r.wrappers.NodeMap.PythonNodeMap(data)
    graph = r.wrappers.Graph.NetworkXGraph(networkx_graph)

    MultiVerify(r).compute(
        "centrality.pagerank",
        graph,
        tolerance=1e-7
    ).assert_equal(expected_val, rel_tol=1e-5)


Calling ``.assert_equal()`` with the expected value and any parameters affecting the comparison accuracy
will perform the assertions. If no ``AssertionError`` is raised, then the results from all concrete algorithms
match the expected value.

If any fail, an error is raised with additional information of which algorithm produced the failing results.

custom_compare
~~~~~~~~~~~~~~

Sometimes, ``assert_equal`` is insufficient for verifying that multiple concrete algorithms have the same
behavior. ``custom_compare`` gives the user full flexibility over how to compare results which by nature are
non-deterministic.

Consider the `Louvain community detection algorithm <https://en.wikipedia.org/wiki/Louvain_modularity>`_. This algorithm
attempts to find communities in a graph that minimize a modularity metric, but includes elements of randomness in
the solution. Thus, simply checking for the same community label assignments for each node in a node map is insufficient.

Here's an example of how ``custom_compare`` might be used to verify reasonable correctness:

.. code-block:: python

    import metagraph as mg
    from metagraph.core.multiverify import MultiVerify

    r = mg.resolver
    ebunch = [
        (0, 3, 1),
        (1, 0, 2),
        (1, 4, 3),
        (2, 5, 5),
        (2, 7, 6),
        (3, 1, 7),
        (3, 4, 8),
        (5, 6, 10),
        (6, 2, 11),
    ]
    nx_graph = nx.Graph()
    nx_graph.add_weighted_edges_from(ebunch)
    graph = r.wrappers.Graph.NetworkXGraph(nx_graph)

    def cmp_func(x):
        x_graph, modularity_score = x
        assert x_graph.num_nodes == 8, x_graph.num_nodes
        assert modularity_score > 0.45

    results = MultiVerify(r).compute("clustering.louvain_community", graph)
    results.normalize(r.types.NodeMap.PythonNodeMap).custom_compare(cmp_func)

``custom_compare`` takes a comparison function (in this example ``cmp_func``). The comparison function is passed the output
of each concrete algorithm and verifies expected behavior.

To ensure that the comparison function only has to deal with a single type, ``normalize`` is typically called prior
to calling ``custom_compare``. In this case, the normalization is not strictly necessary as all ``NodeMap`` objects
have a ``.num_nodes`` property.

In this example, ``cmp_func`` simply takes the modularity score and verifies that it is above a selected threshold.

transform
~~~~~~~~~

While ``assert_equal`` is for exact matches and ``custom_compare`` gives full flexibility, ``transform`` provides
a hybrid solution to the problem of non-deterministic results.

Often, while the solution is non-deterministic, there are elements of the solution which will be deterministic.
Consider a max-flow problem. At the bottlenecks, the flow into a node will be consistent for any correct solution.
Thus, if we can remove all other nodes and simply compare values for the bottleneck nodes, we could use the simpler
``assert_equal`` method. That is where ``transform`` comes in to play.

``transform`` behaves nearly identically to ``compute`` with two key differences:

1. At least one arg of kwarg must be a normalized ``MultiResult``.
2. The first argument is not the path to an abstract algorithm. It must be an exact algorithm call.

The reason for these differences is that we are not trying to exercise all concrete algorithms to check for correctness.
Instead, we simply want to run all the results through the same algorithm in order to simplify the result in
preparation for a call to ``assert_equal`` or ``custom_compare``.

Multiple transforms can be performed on the results, chained together sequentially.

Here is an example for max flow:

.. code-block:: python

    r = mg.resolver
    nx_graph = nx.DiGraph()
    nx_graph.add_weighted_edges_from([
        (0, 1, 9),
        (0, 3, 10),
        (1, 4, 3),
        (2, 7, 6),
        (3, 1, 2),
        (3, 4, 8),
        (4, 5, 7),
        (4, 2, 4),
        (5, 2, 5),
        (5, 6, 1),
        (6, 2, 11),
    ])
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)

    # These are the elements of the result which *are* deterministic
    expected_flow_value = 6
    bottleneck_nodes = dpr.wrappers.NodeSet.PythonNodeSet({2, 4})
    expected_nodemap = dpr.wrappers.NodeMap.PythonNodeMap({2: 6, 4: 6})

    mv = MultiVerify(dpr)
    results = mv.compute("flow.max_flow", graph, source_node=0, target_node=7)

    # Note: each algorithm returns a tuple of (flow_rate, graph_of_flow_values)

    # Compare flow rate
    results[0].assert_equal(expected_flow_value)

    # Normalize actual flow to prepare to transform
    actual_flow = results[1].normalize(dpr.wrappers.Graph.NetworkXGraph)

    # Compare sum of out edges for bottleneck nodes
    out_edges = mv.transform(
        dpr.plugins.core_networkx.algos.util.graph.aggregate_edges,
        actual_flow,
        lambda x, y: x + y,
        initial_value=0,
    )
    out_bottleneck = mv.transform(
        dpr.plugins.core_python.algos.util.nodemap.select, out_edges, bottleneck_nodes
    )
    out_bottleneck.assert_equal(expected_nodemap)

The result from max flow is a tuple of (flow_rate, graph_of_flow_values). The flow rate is compared first by
indexing into the results and using ``assert_equal`` with the expected flow value.

The graph of flow values is first normalized, then transformed by aggregating edges and another transformation
to filter to only keep the bottleneck nodes. At this point, the results are deterministic and can be compared
using ``assert_equal``.

The workflow for each concrete algorithm was:

1. Compute max flow
2. The flow rate was compared against the expected value
3. The flow graph was normalized to a networkx graph
4. The flow graph was translated to a node map using the core_networkx plugin's version of "graph.aggregate_edges"
5. The node map was filtered using the core_python plugin's version of "nodemap.select"
6. The smaller node map was compared against the expected result

This comparison could have been done using ``custom_compare`` and manually calculating the flow rate out of
the bottleneck nodes, but using ``transform`` allows easy access to existing
utility algorithms which are often adequate to extract the deterministic portions and compare using ``assert_equal``.
