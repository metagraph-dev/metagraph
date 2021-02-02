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
  - `scikit-learn <https://scikit-learn.org/stable/>`_

.. raw:: html

   <details>
   <summary>Results from running the test suite</summary>

.. code-block::

    >>> pytest
    =========================================================================================================== test session starts ===========================================================================================================
    platform linux -- Python 3.7.9, pytest-6.2.2, py-1.10.0, pluggy-0.13.1
    rootdir: /home/pnguyen/code/metagraph, configfile: setup.cfg, testpaths: metagraph/tests
    plugins: cov-2.11.1
    collected 198 items                                                                                                                                                                                                                       
    
    metagraph/tests/test_config.py .                                                                                                                                                                                                    [  0%]
    metagraph/tests/test_dask.py .....                                                                                                                                                                                                  [  3%]
    metagraph/tests/test_dtypes.py .                                                                                                                                                                                                    [  3%]
    metagraph/tests/test_entrypoints.py ...                                                                                                                                                                                             [  5%]
    metagraph/tests/test_explorer.py ..........                                                                                                                                                                                         [ 10%]
    metagraph/tests/test_multiverify.py .........                                                                                                                                                                                       [ 14%]
    metagraph/tests/test_node_labels.py ...                                                                                                                                                                                             [ 16%]
    metagraph/tests/test_plugin.py .........                                                                                                                                                                                            [ 20%]
    metagraph/tests/test_registry.py ..                                                                                                                                                                                                 [ 21%]
    metagraph/tests/test_resolver.py ..............................                                                                                                                                                                     [ 36%]
    metagraph/tests/test_toplevel.py ...                                                                                                                                                                                                [ 38%]
    metagraph/tests/test_typecache.py .                                                                                                                                                                                                 [ 38%]
    metagraph/tests/test_types.py ...                                                                                                                                                                                                   [ 40%]
    metagraph/tests/algorithms/test_bipartite.py .                                                                                                                                                                                      [ 40%]
    metagraph/tests/algorithms/test_centrality.py .........                                                                                                                                                                             [ 45%]
    metagraph/tests/algorithms/test_clustering.py .......                                                                                                                                                                               [ 48%]
    metagraph/tests/algorithms/test_embedding.py ssssss.                                                                                                                                                                                [ 52%]
    metagraph/tests/algorithms/test_flow.py ..                                                                                                                                                                                          [ 53%]
    metagraph/tests/algorithms/test_subgraph.py ....s.....                                                                                                                                                                              [ 58%]
    metagraph/tests/algorithms/test_traversal.py ..........                                                                                                                                                                             [ 63%]
    metagraph/tests/algorithms/test_utility.py ...............s                                                                                                                                                                         [ 71%]
    metagraph/tests/translators/test_bipartite.py ..                                                                                                                                                                                    [ 72%]
    metagraph/tests/translators/test_dataframe.py .                                                                                                                                                                                     [ 73%]
    metagraph/tests/translators/test_edgemap.py .......                                                                                                                                                                                 [ 76%]
    metagraph/tests/translators/test_edgeset.py ...                                                                                                                                                                                     [ 78%]
    metagraph/tests/translators/test_graph.py ................                                                                                                                                                                          [ 86%]
    metagraph/tests/translators/test_matrix.py ...                                                                                                                                                                                      [ 87%]
    metagraph/tests/translators/test_node_map.py ....                                                                                                                                                                                   [ 89%]
    metagraph/tests/translators/test_node_set.py ....                                                                                                                                                                                   [ 91%]
    metagraph/tests/translators/test_vector.py ..                                                                                                                                                                                       [ 92%]
    metagraph/tests/types/test_dataframe.py .                                                                                                                                                                                           [ 93%]
    metagraph/tests/types/test_edges.py ...                                                                                                                                                                                             [ 94%]
    metagraph/tests/types/test_graph.py ...                                                                                                                                                                                             [ 96%]
    metagraph/tests/types/test_matrix.py ..                                                                                                                                                                                             [ 97%]
    metagraph/tests/types/test_nodes.py ...                                                                                                                                                                                             [ 98%]
    metagraph/tests/types/test_vector.py ..                                                                                                                                                                                             [100%]
    
    ----------- coverage: platform linux, python 3.7.9-final-0 -----------
    Name                                              Stmts   Miss  Cover   Missing
    -------------------------------------------------------------------------------
    metagraph/__init__.py                                39      0   100%
    metagraph/core/__init__.py                            0      0   100%
    metagraph/core/dask/__init__.py                       0      0   100%
    metagraph/core/dask/placeholder.py                   75     16    79%   24-25, 31, 35, 44-49, 79, 89-94
    metagraph/core/dask/resolver.py                     160     33    79%   93-95, 98, 101, 124-126, 167, 190, 195-214, 233, 241, 246, 256, 270-271, 275-276
    metagraph/core/dtypes.py                             26      0   100%
    metagraph/core/entrypoints.py                        21      0   100%
    metagraph/core/exceptions.py                          2      0   100%
    metagraph/core/multiverify.py                       221     15    93%   15-16, 22, 166, 210-219, 328, 370, 375-379, 402, 405
    metagraph/core/node_labels.py                        49      7    86%   46, 48, 58, 60, 65, 72, 76
    metagraph/core/planning.py                          271     47    83%   76, 82-86, 91-92, 96-97, 101-105, 124, 198, 215, 232-233, 293, 300, 315, 338-357, 363, 375-378, 384, 392, 397-401, 415-416, 421-425, 428
    metagraph/core/plugin.py                            295     24    92%   36, 55, 125, 129, 133, 140, 167, 208, 215, 233, 241, 286, 299, 324, 341, 390, 411, 442, 445, 473, 519, 521, 621, 625
    metagraph/core/plugin_registry.py                    68      0   100%
    metagraph/core/resolver.py                          657     49    93%   73-79, 198-206, 245, 274, 303, 329, 404-405, 409-410, 446, 458, 468, 523, 649, 654, 758, 808, 856, 860, 873, 914, 1008, 1083, 1156-1164, 1167, 1180, 1182, 1223, 1229-1232, 1293-1294
    metagraph/core/roundtrip.py                         115      5    96%   123, 130, 206, 215, 222
    metagraph/core/typecache.py                          82      8    90%   19, 22-25, 85, 89, 100
    metagraph/core/typing.py                             76      9    88%   55, 74, 77-80, 90, 122, 127
    metagraph/dask.py                                     4      0   100%
    metagraph/explorer/__init__.py                        1      0   100%
    metagraph/explorer/api.py                           245     34    86%   27, 36, 48, 146, 206, 283-289, 293-298, 309-310, 334, 336, 338, 360-369, 382, 397-398
    metagraph/explorer/service.py                       109    109     0%   1-169
    metagraph/plugins/__init__.py                        35      2    94%   39-42
    metagraph/plugins/core/__init__.py                    1      0   100%
    metagraph/plugins/core/algorithms/__init__.py         1      0   100%
    metagraph/plugins/core/algorithms/bipartite.py        4      0   100%
    metagraph/plugins/core/algorithms/centrality.py      18      0   100%
    metagraph/plugins/core/algorithms/clustering.py      17      0   100%
    metagraph/plugins/core/algorithms/embedding.py       20      0   100%
    metagraph/plugins/core/algorithms/flow.py             8      0   100%
    metagraph/plugins/core/algorithms/subgraph.py        21      0   100%
    metagraph/plugins/core/algorithms/traversal.py       21      0   100%
    metagraph/plugins/core/algorithms/utility.py         34      0   100%
    metagraph/plugins/core/exceptions.py                  5      0   100%
    metagraph/plugins/core/types.py                      36      0   100%
    metagraph/plugins/core/wrappers.py                   17      0   100%
    metagraph/plugins/graphblas/__init__.py               1      0   100%
    metagraph/plugins/graphblas/algorithms.py           102      4    96%   131-134
    metagraph/plugins/graphblas/translators.py           94      2    98%   136, 153
    metagraph/plugins/graphblas/types.py                223     26    88%   80, 83, 117, 120, 123, 259-274, 309, 315, 344-345, 472-474, 508-509, 520
    metagraph/plugins/networkx/__init__.py                1      0   100%
    metagraph/plugins/networkx/algorithms.py            353     26    93%   110, 418, 426-427, 432-434, 448, 457-458, 463-464, 480, 489-490, 495-497, 526, 530, 537-538, 543-544, 563, 568
    metagraph/plugins/networkx/translators.py            27      0   100%
    metagraph/plugins/networkx/types.py                 211     27    87%   12, 38, 184, 190, 196, 224-226, 241-243, 312-322, 325-335
    metagraph/plugins/numpy/__init__.py                   1      0   100%
    metagraph/plugins/numpy/algorithms.py                47      0   100%
    metagraph/plugins/numpy/translators.py               41      0   100%
    metagraph/plugins/numpy/types.py                    153     27    82%   58, 60, 66-69, 87-88, 91, 94-98, 136, 138, 146, 150, 159, 181-182, 185-189, 195-198
    metagraph/plugins/pandas/__init__.py                  1      0   100%
    metagraph/plugins/pandas/algorithms.py               12      0   100%
    metagraph/plugins/pandas/translators.py              36      0   100%
    metagraph/plugins/pandas/types.py                   132     10    92%   34-35, 69, 75-76, 79, 186-187, 190, 226
    metagraph/plugins/python/__init__.py                  1      0   100%
    metagraph/plugins/python/algorithms.py               30      0   100%
    metagraph/plugins/python/translators.py              22      0   100%
    metagraph/plugins/python/types.py                    39      1    97%   47
    metagraph/plugins/scipy/__init__.py                   1      0   100%
    metagraph/plugins/scipy/algorithms.py               186      3    98%   75-76, 86
    metagraph/plugins/scipy/translators.py               98      1    99%   103
    metagraph/plugins/scipy/types.py                    182      8    96%   28, 40, 118-121, 125, 232-233, 347
    metagraph/tests/__init__.py                           0      0   100%
    metagraph/tests/__main__.py                           7      7     0%   1-10
    metagraph/tests/algorithms/__init__.py                1      0   100%
    metagraph/tests/algorithms/test_bipartite.py         15      0   100%
    metagraph/tests/algorithms/test_centrality.py        82      0   100%
    metagraph/tests/algorithms/test_clustering.py        85      0   100%
    metagraph/tests/algorithms/test_embedding.py        267    136    49%   36-51, 86-105, 145-167, 232-254, 310-422, 459-489
    metagraph/tests/algorithms/test_flow.py              42      0   100%
    metagraph/tests/algorithms/test_subgraph.py         131      2    98%   103, 174
    metagraph/tests/algorithms/test_traversal.py        137      0   100%
    metagraph/tests/algorithms/test_utility.py          181      0   100%
    metagraph/tests/bad_site_dir/__init__.py              0      0   100%
    metagraph/tests/bad_site_dir2/__init__.py             0      0   100%
    metagraph/tests/conftest.py                           4      0   100%
    metagraph/tests/plugins/__init__.py                   0      0   100%
    metagraph/tests/site_dir/__init__.py                  0      0   100%
    metagraph/tests/site_dir/plugin1.py                  14      0   100%
    metagraph/tests/test_config.py                        7      0   100%
    metagraph/tests/test_dask.py                         81      0   100%
    metagraph/tests/test_dtypes.py                        8      0   100%
    metagraph/tests/test_entrypoints.py                  20      0   100%
    metagraph/tests/test_explorer.py                     64      0   100%
    metagraph/tests/test_multiverify.py                 125      2    98%   134-135
    metagraph/tests/test_node_labels.py                  36      0   100%
    metagraph/tests/test_plugin.py                      106      0   100%
    metagraph/tests/test_registry.py                     32      0   100%
    metagraph/tests/test_resolver.py                    581      0   100%
    metagraph/tests/test_toplevel.py                     21      0   100%
    metagraph/tests/test_typecache.py                    28      0   100%
    metagraph/tests/test_types.py                        20      0   100%
    metagraph/tests/translators/__init__.py               1      0   100%
    metagraph/tests/translators/test_bipartite.py        29      0   100%
    metagraph/tests/translators/test_dataframe.py         8      0   100%
    metagraph/tests/translators/test_edgemap.py          43      0   100%
    metagraph/tests/translators/test_edgeset.py          16      0   100%
    metagraph/tests/translators/test_graph.py           201      0   100%
    metagraph/tests/translators/test_matrix.py           28      0   100%
    metagraph/tests/translators/test_node_map.py         68      0   100%
    metagraph/tests/translators/test_node_set.py         31      0   100%
    metagraph/tests/translators/test_vector.py           21      0   100%
    metagraph/tests/types/__init__.py                     0      0   100%
    metagraph/tests/types/test_dataframe.py              10      0   100%
    metagraph/tests/types/test_edges.py                  74      0   100%
    metagraph/tests/types/test_graph.py                 103      0   100%
    metagraph/tests/types/test_matrix.py                 19      0   100%
    metagraph/tests/types/test_nodes.py                  33      0   100%
    metagraph/tests/types/test_vector.py                 19      0   100%
    metagraph/tests/util.py                             129     10    92%   56-57, 107, 124-125, 139, 143, 145-146, 162
    -------------------------------------------------------------------------------
    TOTAL                                              7676    650    92%
    Coverage HTML written to dir htmlcov
    
    
    ========================================================================================================= short test summary info =========================================================================================================
    SKIPPED [1] metagraph/core/multiverify.py:202: No concrete algorithms exist which implement embedding.train.node2vec
    SKIPPED [1] metagraph/core/multiverify.py:202: No concrete algorithms exist which implement embedding.train.graph2vec
    SKIPPED [1] metagraph/core/multiverify.py:202: No concrete algorithms exist which implement embedding.train.graphwave
    SKIPPED [1] metagraph/core/multiverify.py:202: No concrete algorithms exist which implement embedding.train.hope.katz
    SKIPPED [1] metagraph/tests/algorithms/test_embedding.py:308: metagraph_stellargraph not installed.
    SKIPPED [1] metagraph/core/multiverify.py:202: No concrete algorithms exist which implement embedding.train.line
    SKIPPED [1] metagraph/core/multiverify.py:202: No concrete algorithms exist which implement subgraph.subisomorphic
    SKIPPED [1] metagraph/core/multiverify.py:202: No concrete algorithms exist which implement util.graph.isomorphic
    ===================================================================================================== 190 passed, 8 skipped in 35.97s =====================================================================================================
.. raw:: html

   </details>
