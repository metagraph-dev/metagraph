Installation 
============

Metagraph is a pure-python library, making it easy to install with ``pip`` or ``conda``.

However, Metagraph interfaces with many extension libraries where installation using ``pip``
can be more challenging. For this reason, we recommend installing using ``conda``.


Python version support
----------------------

Python 3.8 and above is supported


Installing using conda
----------------------

::

    conda install -c conda-forge -c metagraph metagraph


Installing from PyPI
--------------------

::

    pip install metagraph


Required Dependencies
---------------------

These should be automatically installed when metagraph is installed

  - `numpy <https://numpy.org>`__
  - `scipy <https://scipy.org>`__
  - `dask <https://dask.org/>`__
  - `donfig <https://donfig.readthedocs.io/>`__


Recommended Dependencies
------------------------

Beyond the required dependencies above, additional dependencies are recommended for interchange
formats and basic algorithm coverage.

  - `networkx <https://networkx.github.io/>`_
  - `pandas <https://pandas.pydata.org/>`_
  - `grblas <https://github.com/metagraph-dev/grblas/>`_

A list of additional plugins to provide type and algorithm coverage for more libraries and hardware
can be found in the list of :ref:`community plugins<existing_plugins>`.


Running the test suite
----------------------

Running tests requires additional libraries to be installed. The easiest way to run tests is to install
the ``metagraph-dev`` package from the ``metagraph`` channel.

::

    conda install -c metagraph metagraph-dev


.. raw:: html

   <details>
   <summary>Results from running the test suite</summary>

.. code-block::

    >>> pytest
    =============================== test session starts ================================
    platform darwin -- Python 3.8.8, pytest-6.2.2, py-1.10.0, pluggy-0.13.1
    rootdir: /Projects/metagraph, configfile: setup.cfg, testpaths: metagraph/tests
    plugins: cov-2.8.1
    collected 248 items

    metagraph/tests/test_config.py .                                              [  0%]
    metagraph/tests/test_dask.py ..........                                       [  4%]
    metagraph/tests/test_dask_visualize.py ....                                   [  6%]
    metagraph/tests/test_dtypes.py .                                              [  6%]
    metagraph/tests/test_entrypoints.py ...                                       [  7%]
    metagraph/tests/test_explorer.py ...........                                  [ 12%]
    metagraph/tests/test_multiverify.py ..........                                [ 16%]
    metagraph/tests/test_node_labels.py ....                                      [ 17%]
    metagraph/tests/test_plugin.py ..........                                     [ 21%]
    metagraph/tests/test_registry.py ..                                           [ 22%]
    metagraph/tests/test_resolver.py .................................            [ 35%]
    metagraph/tests/test_resolver_planning.py .....                               [ 37%]
    metagraph/tests/test_roundtrip.py .                                           [ 38%]
    metagraph/tests/test_toplevel.py ...                                          [ 39%]
    metagraph/tests/test_typecache.py ..                                          [ 40%]
    metagraph/tests/test_types.py ..                                              [ 41%]
    metagraph/tests/test_typing.py .....                                          [ 43%]
    metagraph/tests/algorithms/test_bipartite.py .                                [ 43%]
    metagraph/tests/algorithms/test_centrality.py .........                       [ 47%]
    metagraph/tests/algorithms/test_clustering.py .......                         [ 50%]
    metagraph/tests/algorithms/test_embedding.py ssssss.                          [ 52%]
    metagraph/tests/algorithms/test_flow.py ..                                    [ 53%]
    metagraph/tests/algorithms/test_subgraph.py ....s......                       [ 58%]
    metagraph/tests/algorithms/test_traversal.py ..........                       [ 62%]
    metagraph/tests/algorithms/test_utility.py ...............s                   [ 68%]
    metagraph/tests/compiler/test_compile.py ..                                   [ 69%]
    metagraph/tests/compiler/test_plugin.py ....                                  [ 70%]
    metagraph/tests/compiler/test_subgraphs.py ...............                    [ 77%]
    metagraph/tests/translators/test_bipartite.py ..                              [ 77%]
    metagraph/tests/translators/test_dataframe.py .                               [ 78%]
    metagraph/tests/translators/test_edgemap.py .......                           [ 81%]
    metagraph/tests/translators/test_edgeset.py ...                               [ 82%]
    metagraph/tests/translators/test_graph.py ................                    [ 88%]
    metagraph/tests/translators/test_matrix.py ...                                [ 89%]
    metagraph/tests/translators/test_node_map.py ....                             [ 91%]
    metagraph/tests/translators/test_node_set.py ....                             [ 93%]
    metagraph/tests/translators/test_vector.py ..                                 [ 93%]
    metagraph/tests/types/test_bipartite.py .                                     [ 94%]
    metagraph/tests/types/test_dataframe.py .                                     [ 94%]
    metagraph/tests/types/test_edges.py ...                                       [ 95%]
    metagraph/tests/types/test_graph.py ...                                       [ 97%]
    metagraph/tests/types/test_matrix.py ..                                       [ 97%]
    metagraph/tests/types/test_nodes.py ...                                       [ 99%]
    metagraph/tests/types/test_vector.py ..                                       [100%]

    ---------- coverage: platform darwin, python 3.8.8-final-0 -----------
    Name                                 Stmts   Miss  Cover   Missing
    ------------------------------------------------------------------
    metagraph/core/dask/placeholder.py      94      7    93%   48-53, 121
    metagraph/core/dask/resolver.py        157     19    88%   99, 102, 200-219, 253, 258
    metagraph/core/dask/tasks.py            48      4    92%   15, 67-68, 75
    metagraph/core/dask/visualize.py        48      3    94%   40, 70, 76
    metagraph/core/multiverify.py          219      5    98%   22, 166, 210-219
    metagraph/core/roundtrip.py            118      2    98%   117-119
    metagraph/explorer/service.py           79      7    91%   16-17, 79-83
    ------------------------------------------------------------------
    TOTAL                                 4980     47    99%

    55 files skipped due to complete coverage.

    =================== 240 passed, 8 skipped, 2 warnings in 18.06s ====================
.. raw:: html

   </details>
