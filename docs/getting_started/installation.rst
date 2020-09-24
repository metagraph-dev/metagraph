Installation 
============

Metagraph is a pure-python library, making it easy to install with ``pip`` or ``conda``.

However, Metagraph interfaces with many extension libraries where installation using ``pip``
can be more challenging. For this reason, we recommend installing using ``conda``.


Python version support
----------------------

Python 3.7 and above is supported


Installing using conda
----------------------

::

    conda install -c conda-forge metagraph


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
  - `importlib_metadata <https://importlib-metadata.readthedocs.io/>`__
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

Running tests requires additional libraries to be installed

  - `pytest <https://docs.pytest.org/>`_
  - `coverage <https://coverage.readthedocs.io/>`_
  - `pytest-cov <https://pytest-cov.readthedocs.io/>`_

.. raw:: html

   <details>
   <summary>Results from running the test suite</summary>

.. code-block::

    >>> pytest
    ================================================================ test session starts ================================================================
    platform darwin -- Python 3.7.7, pytest-5.4.2, py-1.9.0, pluggy-0.13.1
    rootdir: /projects/metagraph, inifile: setup.cfg, testpaths: metagraph/tests
    plugins: cov-2.8.1
    collected 125 items

    metagraph/tests/test_config.py .                                                                                                              [  0%]
    metagraph/tests/test_dask.py ...                                                                                                              [  3%]
    metagraph/tests/test_dtypes.py .                                                                                                              [  4%]
    metagraph/tests/test_entrypoints.py ...                                                                                                       [  6%]
    metagraph/tests/test_multiverify.py .........                                                                                                 [ 13%]
    metagraph/tests/test_node_labels.py ...                                                                                                       [ 16%]
    metagraph/tests/test_plugin.py .........                                                                                                      [ 23%]
    metagraph/tests/test_registry.py ..                                                                                                           [ 24%]
    metagraph/tests/test_resolver.py ............................                                                                                 [ 47%]
    metagraph/tests/test_toplevel.py ...                                                                                                          [ 49%]
    metagraph/tests/test_typecache.py .                                                                                                           [ 50%]
    metagraph/tests/test_types.py ...                                                                                                             [ 52%]
    metagraph/tests/algorithms/test_betweenness_centrality.py ..                                                                                  [ 54%]
    metagraph/tests/algorithms/test_bipartite.py .                                                                                                [ 55%]
    metagraph/tests/algorithms/test_clustering.py ....                                                                                            [ 58%]
    metagraph/tests/algorithms/test_flow.py .                                                                                                     [ 59%]
    metagraph/tests/algorithms/test_katz_centrality.py .                                                                                          [ 60%]
    metagraph/tests/algorithms/test_pagerank.py .                                                                                                 [ 60%]
    metagraph/tests/algorithms/test_subgraph.py ..                                                                                                [ 62%]
    metagraph/tests/algorithms/test_traversal.py ......                                                                                           [ 67%]
    metagraph/tests/algorithms/test_triangle_count.py .                                                                                           [ 68%]
    metagraph/tests/algorithms/test_utility.py .............                                                                                      [ 78%]
    metagraph/tests/translators/test_graph.py ...                                                                                                 [ 80%]
    metagraph/tests/translators/test_matrix.py ..                                                                                                 [ 82%]
    metagraph/tests/translators/test_node_map.py ...                                                                                              [ 84%]
    metagraph/tests/translators/test_node_set.py ...                                                                                              [ 87%]
    metagraph/tests/translators/test_vector.py .                                                                                                  [ 88%]
    metagraph/tests/types/test_dataframe.py .                                                                                                     [ 88%]
    metagraph/tests/types/test_edges.py ...                                                                                                       [ 91%]
    metagraph/tests/types/test_graph.py ..                                                                                                        [ 92%]
    metagraph/tests/types/test_matrix.py ...                                                                                                      [ 95%]
    metagraph/tests/types/test_nodes.py ....                                                                                                      [ 98%]
    metagraph/tests/types/test_vector.py ..                                                                                                       [100%]

    ================================================================= warnings summary ==================================================================
    /miniconda3/envs/mg/lib/python3.7/site-packages/donfig/config_obj.py:30
      /miniconda3/envs/mg/lib/python3.7/site-packages/donfig/config_obj.py:30: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3,and in 3.9 it will stop working
        from collections import Mapping

    -- Docs: https://docs.pytest.org/en/latest/warnings.html

    ---------- coverage: platform darwin, python 3.7.7-final-0 -----------
    Name                                                        Stmts   Miss  Cover   Missing
    -----------------------------------------------------------------------------------------
    metagraph/__init__.py                                          40      0   100%
    metagraph/algorithms/__init__.py                                1      0   100%
    metagraph/algorithms/bipartite.py                               4      0   100%
    metagraph/algorithms/centrality.py                              9      0   100%
    metagraph/algorithms/clustering.py                             15      0   100%
    metagraph/algorithms/flow.py                                    6      0   100%
    metagraph/algorithms/subgraph.py                                6      0   100%
    metagraph/algorithms/traversal.py                              15      0   100%
    metagraph/algorithms/utility.py                                28      0   100%
    metagraph/core/__init__.py                                      0      0   100%
    metagraph/core/dask/__init__.py                                 0      0   100%
    metagraph/core/dask/placeholder.py                             72     19    74%   14, 18, 27-32, 46, 62, 76-85, 111
    metagraph/core/dask/resolver.py                               144     29    80%   92-94, 117-119, 160, 183, 188-207, 218-223, 226, 234, 247
    metagraph/core/dtypes.py                                       26      0   100%
    metagraph/core/entrypoints.py                                  21      0   100%
    metagraph/core/multiverify.py                                 178      1    99%   227
    metagraph/core/node_labels.py                                  49      7    86%   46, 48, 58, 60, 65, 72, 76
    metagraph/core/planning.py                                    255     37    85%   23, 29-33, 38-39, 43-44, 48-52, 71, 191, 208, 225-226, 286, 293, 308, 333, 345-348, 354, 362, 367-371, 385-386, 391-395, 398
    metagraph/core/plugin.py                                      273     20    93%   36, 55, 125, 129, 133, 140, 191, 198, 216, 224, 259, 272, 304, 353, 374, 407, 410, 437, 472, 572
    metagraph/core/plugin_registry.py                              68      0   100%
    metagraph/core/resolver.py                                    537     46    91%   62-68, 159-167, 254, 259, 270, 313, 375, 379, 390, 427, 450, 513, 567, 604, 617, 619, 717, 792-793, 817-818, 859, 867-881, 964-965, 969-970
    metagraph/core/typecache.py                                    47      5    89%   19, 22-25
    metagraph/core/typing.py                                       50      7    86%   34, 53, 56-59, 69
    metagraph/dask.py                                               4      0   100%
    metagraph/explorer/__init__.py                                  1      1     0%   1
    metagraph/explorer/api.py                                     243    243     0%   1-473
    metagraph/explorer/service.py                                 109    109     0%   1-169
    metagraph/plugins/__init__.py                                  32      0   100%
    metagraph/plugins/graphblas/__init__.py                         1      0   100%
    metagraph/plugins/graphblas/algorithms.py                      39      0   100%
    metagraph/plugins/graphblas/translators.py                     93     13    86%   23-26, 30-33, 48-51, 122, 125, 132
    metagraph/plugins/graphblas/types.py                          169     27    84%   83, 86, 102-111, 120, 127, 194-195, 246-253, 268-279, 311, 317, 357
    metagraph/plugins/networkx/__init__.py                          1      0   100%
    metagraph/plugins/networkx/algorithms.py                      143      0   100%
    metagraph/plugins/networkx/translators.py                      36     14    61%   28-29, 40-56
    metagraph/plugins/networkx/types.py                           211    107    49%   11, 32, 64, 91, 95, 128-137, 172, 178, 184, 193-272, 287-336
    metagraph/plugins/numpy/__init__.py                             1      0   100%
    metagraph/plugins/numpy/algorithms.py                          88     30    66%   24, 35-42, 54, 62-79, 90-96, 106-112
    metagraph/plugins/numpy/translators.py                         55      3    95%   59-61
    metagraph/plugins/numpy/types.py                              255     51    80%   31, 42-46, 49, 52-56, 59, 62-65, 93, 95, 119, 121, 127, 200, 209, 212, 224, 234-242, 253-256, 260, 262-264, 270-273, 354, 362, 364, 371-372, 392-394, 427
    metagraph/plugins/pandas/__init__.py                            1      0   100%
    metagraph/plugins/pandas/algorithms.py                         12      0   100%
    metagraph/plugins/pandas/translators.py                        35      1    97%   9
    metagraph/plugins/pandas/types.py                             110     32    71%   54, 60-61, 64, 76-83, 157-158, 161, 174-202, 226
    metagraph/plugins/python/__init__.py                            1      0   100%
    metagraph/plugins/python/algorithms.py                         30      0   100%
    metagraph/plugins/python/translators.py                        40      9    78%   10, 26-27, 38-44
    metagraph/plugins/python/types.py                              59      5    92%   24, 27, 66, 69, 74
    metagraph/plugins/scipy/__init__.py                             1      0   100%
    metagraph/plugins/scipy/algorithms.py                         125      2    98%   75-76
    metagraph/plugins/scipy/translators.py                        119     21    82%   13-16, 24-27, 55, 128, 162-179
    metagraph/plugins/scipy/types.py                              142     10    93%   33-34, 73, 79, 154-157, 163, 187, 250-251
    metagraph/tests/__init__.py                                     0      0   100%
    metagraph/tests/algorithms/__init__.py                          1      0   100%
    metagraph/tests/algorithms/test_betweenness_centrality.py      26      0   100%
    metagraph/tests/algorithms/test_bipartite.py                   15      0   100%
    metagraph/tests/algorithms/test_clustering.py                  62      0   100%
    metagraph/tests/algorithms/test_densesparse.py                  1      0   100%
    metagraph/tests/algorithms/test_flow.py                        25      0   100%
    metagraph/tests/algorithms/test_katz_centrality.py             13      0   100%
    metagraph/tests/algorithms/test_pagerank.py                    13      0   100%
    metagraph/tests/algorithms/test_subgraph.py                    24      0   100%
    metagraph/tests/algorithms/test_traversal.py                   63      0   100%
    metagraph/tests/algorithms/test_triangle_count.py              10      0   100%
    metagraph/tests/algorithms/test_utility.py                    146      0   100%
    metagraph/tests/bad_site_dir/__init__.py                        0      0   100%
    metagraph/tests/bad_site_dir2/__init__.py                       0      0   100%
    metagraph/tests/plugins/__init__.py                             0      0   100%
    metagraph/tests/site_dir/__init__.py                            0      0   100%
    metagraph/tests/site_dir/plugin1.py                            14      0   100%
    metagraph/tests/test_config.py                                  7      0   100%
    metagraph/tests/test_dask.py                                   62      0   100%
    metagraph/tests/test_dtypes.py                                  8      0   100%
    metagraph/tests/test_entrypoints.py                            20      0   100%
    metagraph/tests/test_multiverify.py                           120      0   100%
    metagraph/tests/test_node_labels.py                            36      0   100%
    metagraph/tests/test_plugin.py                                106      0   100%
    metagraph/tests/test_registry.py                               32      0   100%
    metagraph/tests/test_resolver.py                              517      0   100%
    metagraph/tests/test_toplevel.py                               21      0   100%
    metagraph/tests/test_typecache.py                              28      0   100%
    metagraph/tests/test_types.py                                  20      0   100%
    metagraph/tests/translators/__init__.py                         0      0   100%
    metagraph/tests/translators/test_graph.py                      39      0   100%
    metagraph/tests/translators/test_matrix.py                     29      0   100%
    metagraph/tests/translators/test_node_map.py                   33      0   100%
    metagraph/tests/translators/test_node_set.py                   26      0   100%
    metagraph/tests/translators/test_vector.py                     16      0   100%
    metagraph/tests/types/__init__.py                               0      0   100%
    metagraph/tests/types/test_dataframe.py                        10      0   100%
    metagraph/tests/types/test_edges.py                            78      0   100%
    metagraph/tests/types/test_graph.py                            62      0   100%
    metagraph/tests/types/test_matrix.py                           41      0   100%
    metagraph/tests/types/test_nodes.py                            36      0   100%
    metagraph/tests/types/test_vector.py                           31      0   100%
    metagraph/tests/util.py                                       110      6    95%   56-57, 103, 107, 109-110
    metagraph/types.py                                             40      0   100%
    metagraph/wrappers.py                                          65      0   100%
    -----------------------------------------------------------------------------------------
    TOTAL                                                        6006    855    86%

    ========================================================== 125 passed, 1 warning in 16.60s ==========================================================

.. raw:: html

   </details>