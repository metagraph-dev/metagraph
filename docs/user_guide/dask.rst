.. _dask:

Usage with Dask
===============

`Dask <https://dask.org/>`__ is integrated into Metagraph to allow algorithms and translations to be run in Dask task
graphs, and to accept Dask data structures as input.

Most libraries which integrate Dask attempt to require as few changes to a user's code as possible to
move from standard mode into Dask mode (often simply a new import).

For example, here is basic Numpy code:

\
    .. code-block:: python

        import numpy as np

        a = np.arange(100)
        result = sum(a + 1)
        result

    5050

And here is the same code written using ``dask.array``, the equivalent of a numpy array:

\
    .. code-block:: python

        import dask.array as da

        a = da.arange(100)
        result = sum(a + 1)
        result.compute()

    5050


The DaskResolver
----------------

The primary element of Metagraph that is replaced for usage with Dask is the ``Resolver``. A new ``DaskResolver``
wraps a ``Resolver`` and intercepts translations and algorithm calls, adding them to the task graph rather than
executing them directly.

To access the default Dask resolver, use ``metagraph.dask.resolver``.

Here is basic Metagraph code to translate between types:

\
    .. code-block:: python

        import numpy as np
        import metagraph as mg
        res = mg.resolver

        x = res.wrappers.NodeMap.NumpyNodeMap(np.array([5, 4, 3, 2, 1]))
        y = res.translate(x, res.types.NodeMap.PythonNodeMapType)
        y

    {0: 5, 1: 4, 2: 3, 3: 2, 4: 1}

And here is the same code using the Dask resolver:

\
    .. code-block:: python

        import numpy as np
        import metagraph.dask as mgdask
        dres = mgdask.resolver

        x = dres.wrappers.NodeMap.NumpyNodeMap(np.array([5, 4, 3, 2, 1]))
        y = dres.translate(x, dres.types.NodeMap.PythonNodeMapType)
        y

    <types.PythonNodeMapTypePlaceholder at 0x7fbd6dc78290>

The result is a ``Placeholder`` object which knows that it will be a ``PythonNodeMapType`` object,
but it hasn't been computed yet. To do that, call ``.compute()`` on the object.

\
    .. code-block:: python

        y.compute()

    {0: 5, 1: 4, 2: 3, 3: 2, 4: 1}


Custom Dask Resolvers
~~~~~~~~~~~~~~~~~~~~~

Creating new Resolvers is an advanced feature that most users of Metagraph won't need. However, if you do
need to create a custom ``Resolver``, converting that into its lazy equivalent is easy -- simply wrap it in
``metagraph.dask.DaskResolver``.

\
    .. code-block:: python

        from metagraph.core.resolver import Resolver
        from metagraph.dask import DaskResolver

        custom_resolver = Resolver()
        custom_resolver.register(...)  # register whatever pieces are desired
        lazy_resolver = DaskResolver(custom_resolver)
        # Now `lazy_resolver` has the same registered items, but operates lazily


Placeholders
------------

A ``Placeholder`` is the Dask equivalent of a ``ConcreteType`` and each concrete type will have a corresponding
class in the Dask resolver. The class name is the name of the concrete type with "Placeholder" tacked on as
as suffix.

For example, ``NetworkXGraphType`` has a ``NetworkXGraphTypePlaceholder`` class.

The purpose of Placeholders is to delay computation while still providing information to Metagraph about the
resultant type, allowing further chaining of the delayed computations.

This is an example of chained operations showing how Placeholders function:

\
    .. code-block:: python

        x = dres.wrappers.NodeMap.NumpyNodeMap(np.array([5, 4, 3, 2, 1]))
        y = dres.translate(x, dres.types.NodeMap.PythonNodeMapType)
        print(type(y))
        z = dres.algos.util.nodemap.apply(y, lambda n: n * n)
        print(type(z))
        z.compute()

    | <class 'types.PythonNodeMapTypePlaceholder'>
    | <class 'types.PythonNodeMapTypePlaceholder'>
    | {0: 25, 1: 16, 2: 9, 3: 4, 4: 1}

``y`` is a Placeholder, but Metagraph is able to use it as input to ``util.nodemap.apply`` because the
type is known. Properties are not know at this time, so failure may still occur when the result is computed,
but it allows for the general workflow of translations and algorithm calls to be built into a task graph
via intermediate Placeholder objects.


DelayedWrapper
--------------

In addition to translations and algorithm calls, building of the data objects can also be delayed. Indicating
the resultant type is still a requirement for these delayed objects to work in Metagraph.

A ``DelayedWrapper`` functions similar to ``dask.delayed``, but wraps a constructor and passes in the resultant
type.

As an example, create a delayed constructor for building complete networkx graphs.

\
    .. code-block:: python

        import networkx as nx
        import itertools

        def build_complete_nxgraph(num_nodes):
            g = nx.DiGraph()
            for src, dst in itertools.product(range(num_nodes), range(num_nodes)):
                g.add_edge(src, dst)
            return res.wrappers.Graph.NetworkXGraph(g)

        nx_complete_factory = dres.delayed_wrapper(build_complete_nxgraph, res.types.Graph.NetworkXGraphType)
        print(nx_complete_factory)

    DelayedWrapper<NetworkXGraphType>

``nx_complete_factory`` is a delayed constructor which return objects that are of type ``NetworkXGraphType``.
Calling it using the same signature as the function ``build_complete_nxgraph`` will yield a
``NetworkXGraphTypePlaceholder`` object whose construction has been delayed.

\
    .. code-block:: python

        my_graph = nx_complete_factory(100)
        my_graph

    <types.NetworkXGraphTypePlaceholder at 0x7fbd51122590>

Because ``my_graph`` is a Placeholder, it can be used in algorithm calls and translations by the Dask resolver.


Visualizing the task graph
--------------------------

One very nice benefit of building up a Dask task graph is that Dask comes with builtin visualization features.

Let's take ``my_graph`` from above, translate it, and call an algorithm. Before actually computing anything,
we will visualize the steps Metagraph will take.

\
    .. code-block:: python

        g2 = dres.translate(my_graph, dres.types.Graph.GrblasGraphType)
        pr = dres.algos.centrality.pagerank(g2)
        pr.visualize()

    .. image:: dask_visualize.png

The translation from ``NetworkXGraph`` to ``GrblasGraph`` actually required two steps, so both are represented
in the task graph.

Calling ``pr.compute()`` will perform all of these steps, from building the complete graph to
translating and finally returning the nodemap of pagerank values.