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

The *assert_equal* method of concrete types discussed in :ref:`Plugin Parts<plugin_parts>` comes in handy for translator tests.

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

        # Convert python -> compactnumpy
        intermediate = CompactNumpyNodeMap(np.array([12.5, 33.4, -1.2]), {0: 0, 1: 1, 42: 2})

        y = r.translate(x, CompactNumpyNodeMap)
        r.assert_equal(y, intermediate)

        # Convert python <- compactnumpy
        x2 = r.translate(y, PythonNodeMap)
        r.assert_equal(x, x2)

Here we test translation from a Python node map to a `NumPy <https://numpy.org/>`_ compact node map and back again.

We use the Metagraph resolver's *translate* method to translate as necessary and *assert_equal* method to verify that
the translations are valid. The Metagraph resolver's *assert_equal* method utilizes the *assert_equal* implemented by
the relevant concrete types.

Testing Algorithms
------------------

Since abstract algorithms are merely specifications, we can only test concrete algorithms.

When testing concrete algorithms, simply testing that outputs for given inputs match expectations is insufficient
for verifying that the outputs also match the results from concrete algorithms written in other plugins (that correspond
to the same abstract algorithm).

We highly recommend using the utility *metagraph.tests.algorithms.MultiVerify* as it verifies that all concrete algorithms
for a given abstract algorithm get the same result.

It does this by finding all the concrete algorithms for the given abstract algorithm (known to the given resolver),
using the given resolver's translators to translate the given input types to the appropriate types for every concrete
algorithm, and using the *assert_equal* method of the concrete types to verify that all the results from all the concrete
algorithms are the same.

This additionally also indirectly tests translators.

.. _plugin_testing_multiverify_with_assert_equals:

MultiVerify with assert_equals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's an example of how to use *metagraph.tests.algorithms.MultiVerify*:

.. code-block:: python

    import networkx as nx
    import numpy as np
    import metagraph as mg
    from metagraph.tests.algorithms import MultiVerify

    def test_pagerank(default_plugin_resolver):
        """
              +-+
     ------>  |1|
     |        +-+
     |
     |         |
     |         v

    +-+  <--  +-+       +-+
    |0|       |2|  <--  |3|
    +-+  -->  +-+       +-+
    """
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
        graph = r.wrappers.EdgeMap.NetworkXEdgeMap(networkx_graph)
        MultiVerify(
            r,
            "link_analysis.pagerank",
            graph,
            tolerance=1e-7
        ).assert_equals(expected_val, rel_tol=1e-5)

This is a simple test of `Page Rank <https://en.wikipedia.org/wiki/PageRank>`_.

The first several lines are fairly straightforward set up.

The first noteworthy line is:

.. code-block:: python

    expected_val = r.wrappers.NodeMap.PythonNodeMap(data)

We're generating a Python node map with our expected results.

Next, we generate our input graph.

.. code-block:: python

    graph = r.wrappers.EdgeMap.NetworkXEdgeMap(networkx_graph)

The last line demonstrates how to use *metagraph.tests.algorithms.MultiVerify*:

.. code-block:: python

    MultiVerify(
        r,
        "link_analysis.pagerank",
        graph,
        tolerance=1e-7
    ).assert_equals(expected_val, rel_tol=1e-5)

Note the use of *MultiVerify(r, "link_analysis.pagerank", graph, tolerance=1e-7)*. This generates an instance of the
*MultiVerify* class. The first parameter is the resolver to use. The second parameter is the name of the abstract
algorithm whose concrete algorithms are being tested. The remaining positional and keyword arguments passed into the
*MultiVerify* initializer (in this example, *graph* and *tolerance=1e-7*) are the inputs passed to the concrete
algorithms (the given resolver is used to translate these inputs to the types appropriate for each concrete algorithm).

Once the *MultiVerify* instance is created, the *assert_equals* method of *MultiVerify* is invoked. It takes an expected
value and optionally a relative (via the keyword "rel_tol") and absolute (via the keyword "abs_tol") tolerance. The
relative and absolute tolerances are used to account for minor differences in float values.

Using a *MultiVerify* instance with the *assert_equals* method tests that all of the concrete algorithms known to the
given resolver get the same result. The resolver's translators are used to translate the concrete algorithm inputs to
the necessary type (which indirectly tests translators). This helps sanity check not just one concrete algorithm, but
also sanity checks that all concrete algorithms behave similarly.

MultiVerify with custom_compare
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes, *MultiVerify.assert_equals* is insufficient for verifying that multiple concrete algorithms have the same
behavior.

Consider the `Louvain community detection algorithm <https://en.wikipedia.org/wiki/Louvain_modularity>`_. This algorithm
attempts to find communities in a graph that minimize a modularity metric. This is frequently a computationally
intractable task depending on the modularity metric given. Louvain community detection uses heuristics to minimize
the modularity. Different implementations may yield different community assignments due to non-determinism, random
initialization, parallelism, or a variety of other factors. Thus, simply checking for the same community label
assignments for each node in a node map may be insufficient.

The *custom_compare* method of *MultiVerify* can be useful here.

Here's an example of how to use the *custom_compare* method of *MultiVerify* to test concrete algorithms for Louvain community detection:

.. code-block:: python

    import metagraph as mg
    from metagraph.tests.algorithms import MultiVerify

    def test_louvain(default_plugin_resolver):
        """
    0 ---2-- 1        5 --10-- 6
    |      / |        |      /
    |     /  |        |     /
    1   7    3        5   11
    |  /     |        |  /
    | /      |        | /
    3 --8--- 4        2 --6--- 7
        """
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
        graph = r.wrappers.EdgeMap.NetworkXEdgeMap(nx_graph, weight_label="weight")

        def cmp_func(x):
            x_graph, modularity_score = x
            assert x_graph.num_nodes == 8, x_graph.num_nodes
            assert modularity_score > 0.45

        MultiVerify(r, "clustering.louvain_community", graph).custom_compare(cmp_func)

*custom_compare* takes a comparison function (in this example *cmp_func*). The comparison function is passed the output
of each concrete algorithm and verifies expected behavior.

In this example, *cmp_func* simply takes the modularity score and verifies that it is above a selected threshold.

The *custom_compare* method of *MultiVerify* is useful for cases where concrete algorithms might operate non-deterministically
or that yield approximate results.

Additionally, the *custom_compare* method can also be useful for algorithms that return graphs. Different concrete
algorithms might return isomorphic graphs, but checking for graph isomorphism in general is intractable. Using a custom
compare function can be useful in these cases since a priori knowledge of the expected output graph can make graph
isomorphism checking very fast. For example, if the expected output graph has only one node with 4 out edges, we can
quickly identify the corresponding node.

Suggestions for MultiVerify Extensions
--------------------------------------

If you find that the utilities provided by *MultiVerify* for testing consistent behavior across all concrete algorithm
implementations for a given abstract algorithm are lacking, please let us know `here <https://github.com/ContinuumIO/metagraph/issues>`_.
