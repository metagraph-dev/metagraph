.. _compiler_plugins:

Compiler Plugins
================

Most concrete algorithms in Metagraph are self-contained "black box
implementations" that are run independently by the dispatcher or Dask
scheduler.  However, Metagraph also allows a special kind of concrete
algorithm to be defined which is "compilable."  A compilable algorithm is
described in such a way that a compiler plugin can do just-in-time (JIT)
compilation of the algorithm to a fast, executable form (usually machine
code).  The benefit of JIT compilation is that multiple tasks can be fused
together by the compiler to reduce redundant work and eliminate the Python
call overhead associated with each individual task.

.. note::

    In the future, JIT compilation may also allow specialization of algorithms
    based on input constants.

When using Metagraph with Dask (see :ref:`dask`), a special Dask graph
optimizer is invoked when the ``compute()`` method is called on a
``Placeholder`` object.  This optimizer will scan the Dask task graph for
compilable Metagraph algorithms and identify task subgraphs to send to the
compiler plugin.  Currently, subgraphs must be linear chains of compilable
tasks to ensure that we do not reduce the parallelism of the overall task graph.

.. note::

    Metagraph will someday allow compiler plugins targeting backends with
    parallel execution capabilities to request maximal subgraphs for
    compilation, rather than just linear chains.


A Compiler Example
------------------



Creating a Compiler Plugin
--------------------------

A compiler plugin should have the following form:

.. code-block:: python

    from metagraph.core.plugin import Compiler
    class MyCompiler(Compiler):
        def __init__(self):
            super().__init__(name="example_compiler")

        def initialize_runtime(self):
            # if compiler has any runtime setup requirements, do that here
            # calling this multiple times should have no effect
            pass

        def teardown_runtime(self):
            # shutdown and free resources allocated
            pass

        def compile_algorithm(self, algo:ConcreteAlgorithm, literals:Dict[str,Any])->Callable:
            # compile algo return an equivalent Python callable
            #
            # initialize the runtime if the compiler needs it

        def compile_subgraph(self, subgraph: Dict, inputs: List[Hashable], output: Hashable):
            # compile this Dask subgraph into a single Python callable and return it
            # callable inputs will correspond to the keys in the inputs list
            # and the return value should correspond to the output key.
            #
            # initialize the runtime if the compiler needs it

The name of the compiler is used when defining new concrete algorithms, as
shown in the next section.  Initialization and teardown methods are provided
to defer potentially slow or memory intensive setup until the compiler is
first needed.  Note that `teardown_runtime()` is not currently called by
Metagraph, but is present to make it possible to reinitialize the plugin for
testing purposes.

The `compile_algorithm()` method is used when the user is executing compilable
algorithms immediately from the standard Resolver, wherease the
`compile_subgraph()` method is used when the user is constructing a Dask DAG
using the DaskResolver.  In the latter scenario, subgraphs may have only one
task.
            


Creating a Compilable Concrete Algorithm
----------------------------------------

A compilable concrete algorithm is 


Invoking the Compiler
---------------------

Users will generally not need to interact with or think about the compiler
when using Metagraph.  The optimizer is applied automatically when a Metagraph
placeholder object is computed.  

If you have a larger DAG that uses Metagraph for an intermediate calculation,
you will have to ask Dask to apply the Metagraph optimizer manually.  To do this:

 .. code-block:: python

    import dask
    import metagraph as mg
    # res is a Dask object with internal Metagraph tasks
    res_opt = dask.optimize(res, optimizations=[mg.optimize])
    answer = res_opt.compute()

The Metagraph optimizer will leave all non-Metagraph tasks unchanged, so it is
always safe to apply.


Visualizing Compilation
-----------------------

Metagraph placeholder objects have a custom `visualize()` method which works
the same as the standard `Dask visualize() method`_, but with special shapes
and labels for Metagraph operations.  For example, this DAG:

.. image:: mg_visualize.png

shows translation steps with ellipses, the concrete type of results with
parallelograms, and algorithms with octagons.

As with optimization, the custom Metagraph visualize method can be used with
any Dask object by calling it directly:

.. code-block:: python

    import metagraph as mg

    mg.visualize(my_dask_object)

When the DAG contains compilable tasks, they will be highlighted with a single
red octagon outline:

.. image:: mg_vis_unopt.png

And when the optimizer has compiled and fused tasks, the tasks will be shown
in a double octagon outline with a label listing the algorithms that were fused:

.. image:: mg_vis_opt.png

By default, the visualizer optimizes the graph before drawing it.  To disable
this, pass ``optimize_graph=False`` to the ``visualize()`` method.

    




.. _dask visualize() method: https://docs.dask.org/en/latest/graphviz.html
