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

  - `numpy <https://numpy.org>`_
  - `scipy <https://scipy.org>`_
  - `importlib_metadata <https://importlib-metadata.readthedocs.io/>`_
  - `donfig <https://donfig.readthedocs.io/>`_


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
    =============================================================== test session starts ===============================================================
    platform darwin -- Python 3.7.7, pytest-5.4.2, py-1.8.1, pluggy-0.13.1
    rootdir: /projects/metagraph, inifile: setup.cfg, testpaths: metagraph/tests
    plugins: cov-2.8.1
    collected 84 items

    metagraph/tests/test_config.py .                                                                                                            [  1%]
    metagraph/tests/test_dtypes.py .                                                                                                            [  2%]
    metagraph/tests/test_entrypoints.py ..                                                                                                      [  4%]
    metagraph/tests/test_node_labels.py ...                                                                                                     [  8%]
    metagraph/tests/test_plugin.py .........                                                                                                    [ 19%]
    metagraph/tests/test_registry.py ..                                                                                                         [ 21%]
    metagraph/tests/test_resolver.py ......................                                                                                     [ 47%]
    metagraph/tests/test_toplevel.py ...                                                                                                        [ 51%]
    metagraph/tests/test_typecache.py .                                                                                                         [ 52%]
    metagraph/tests/test_types.py .                                                                                                             [ 53%]
    metagraph/tests/algorithms/test_betweenness_centrality.py .                                                                                 [ 54%]
    metagraph/tests/algorithms/test_clustering.py ....                                                                                          [ 59%]
    metagraph/tests/algorithms/test_katz_centrality.py .                                                                                        [ 60%]
    metagraph/tests/algorithms/test_pagerank.py .                                                                                               [ 61%]
    metagraph/tests/algorithms/test_subgraph.py .                                                                                               [ 63%]
    metagraph/tests/algorithms/test_traversal.py ....                                                                                           [ 67%]
    metagraph/tests/algorithms/test_triangle_count.py .                                                                                         [ 69%]
    metagraph/tests/translators/test_graph.py ...                                                                                               [ 72%]
    metagraph/tests/translators/test_matrix.py ..                                                                                               [ 75%]
    metagraph/tests/translators/test_nodes.py ......                                                                                            [ 82%]
    metagraph/tests/translators/test_vector.py .                                                                                                [ 83%]
    metagraph/tests/types/test_dataframe.py .                                                                                                   [ 84%]
    metagraph/tests/types/test_edges.py ....                                                                                                    [ 89%]
    metagraph/tests/types/test_matrix.py ...                                                                                                    [ 92%]
    metagraph/tests/types/test_nodes.py ....                                                                                                    [ 97%]
    metagraph/tests/types/test_vector.py ..                                                                                                     [100%]

    ================================================================ warnings summary =================================================================
    /miniconda3/envs/mg/lib/python3.7/site-packages/donfig/config_obj.py:30
      /miniconda3/envs/mg/lib/python3.7/site-packages/donfig/config_obj.py:30: DeprecationWarning: Using or importing the ABCs
             from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3,and in 3.9 it will stop working
        from collections import Mapping

    -- Docs: https://docs.pytest.org/en/latest/warnings.html

    ---------- coverage: platform darwin, python 3.7.7-final-0 -----------
    Name                                                        Stmts   Miss  Cover   Missing
    -----------------------------------------------------------------------------------------
    metagraph/__init__.py                                          39      0   100%
    metagraph/algorithms/__init__.py                                1      0   100%
    metagraph/algorithms/clustering.py                             13      0   100%
    metagraph/algorithms/subgraph.py                               14      4    71%   7, 12, 17, 22
    metagraph/algorithms/traversal.py                              15      4    73%   8, 13, 18, 27
    metagraph/algorithms/vertex_ranking.py                          8      0   100%
    metagraph/core/__init__.py                                      0      0   100%
    metagraph/core/dtypes.py                                       26      0   100%
    metagraph/core/entrypoints.py                                  21      1    95%   41
    metagraph/core/node_labels.py                                  49      7    86%   43, 45, 55, 57, 62, 69, 73
    metagraph/core/planning.py                                    173     21    88%   19, 44, 46-50, 113, 140, 165-166, 171, 231-239
    metagraph/core/plugin.py                                      252     21    92%   35, 54, 120, 171, 182, 190, 225, 238, 270, 308, 329, 364, 367, 382-384, 397-400, 452
    metagraph/core/plugin_registry.py                              68      0   100%
    metagraph/core/resolver.py                                    362      7    98%   241, 246, 257, 300, 536, 636-637
    metagraph/core/typecache.py                                    47      1    98%   23
    metagraph/plugins/__init__.py                                  28      0   100%
    metagraph/plugins/graphblas/__init__.py                         1      0   100%
    metagraph/plugins/graphblas/translators.py                     51      6    88%   21-24, 28-31
    metagraph/plugins/graphblas/types.py                          146     32    78%   65-66, 72-78, 86, 141-142, 160-163, 166, 172-179, 183-192, 204, 225, 229
    metagraph/plugins/networkx/__init__.py                          1      0   100%
    metagraph/plugins/networkx/algorithms.py                       79      2    97%   79-80
    metagraph/plugins/networkx/translators.py                      23      1    96%   30
    metagraph/plugins/networkx/types.py                            67      3    96%   41, 69, 73
    metagraph/plugins/numpy/__init__.py                             1      0   100%
    metagraph/plugins/numpy/algorithms.py                           4      0   100%
    metagraph/plugins/numpy/translators.py                         71      8    89%   10-13, 18, 88-90
    metagraph/plugins/numpy/types.py                              151     17    89%   17, 19, 69, 73, 76-79, 125, 134-135, 171, 179, 181, 204-206, 226
    metagraph/plugins/pandas/__init__.py                            1      0   100%
    metagraph/plugins/pandas/translators.py                        17      1    94%   10
    metagraph/plugins/pandas/types.py                              82      6    93%   39-40, 95-96, 117, 121
    metagraph/plugins/python/__init__.py                            1      0   100%
    metagraph/plugins/python/translators.py                        21      1    95%   9
    metagraph/plugins/python/types.py                              45      7    84%   15-16, 20-23, 44
    metagraph/plugins/scipy/__init__.py                             1      0   100%
    metagraph/plugins/scipy/algorithms.py                          20      0   100%
    metagraph/plugins/scipy/translators.py                         39      4    90%   23-26
    metagraph/plugins/scipy/types.py                              113     20    82%   33-34, 61, 63, 70-89, 108, 127, 131
    metagraph/tests/__init__.py                                     0      0   100%
    metagraph/tests/algorithms/__init__.py                         87     37    57%   25, 38-47, 67-83, 87-88, 95-102, 131-133, 144, 150-157, 164
    metagraph/tests/algorithms/test_betweenness_centrality.py      15      0   100%
    metagraph/tests/algorithms/test_clustering.py                  62      0   100%
    metagraph/tests/algorithms/test_densesparse.py                  1      0   100%
    metagraph/tests/algorithms/test_katz_centrality.py             13      0   100%
    metagraph/tests/algorithms/test_pagerank.py                    13      0   100%
    metagraph/tests/algorithms/test_subgraph.py                    13      0   100%
    metagraph/tests/algorithms/test_traversal.py                   41      0   100%
    metagraph/tests/algorithms/test_triangle_count.py              10      0   100%
    metagraph/tests/bad_site_dir/__init__.py                        0      0   100%
    metagraph/tests/plugins/__init__.py                             0      0   100%
    metagraph/tests/site_dir/__init__.py                            0      0   100%
    metagraph/tests/site_dir/plugin1.py                            13      0   100%
    metagraph/tests/test_config.py                                  7      0   100%
    metagraph/tests/test_dtypes.py                                  8      0   100%
    metagraph/tests/test_entrypoints.py                            17      0   100%
    metagraph/tests/test_node_labels.py                            32      0   100%
    metagraph/tests/test_plugin.py                                103      3    97%   52, 143-144
    metagraph/tests/test_registry.py                               32      0   100%
    metagraph/tests/test_resolver.py                              423      9    98%   583, 646, 650, 654, 730, 734, 740, 746, 754
    metagraph/tests/test_toplevel.py                               21      0   100%
    metagraph/tests/test_typecache.py                              28      0   100%
    metagraph/tests/test_types.py                                   8      0   100%
    metagraph/tests/translators/__init__.py                         0      0   100%
    metagraph/tests/translators/test_graph.py                      40      0   100%
    metagraph/tests/translators/test_matrix.py                     29      0   100%
    metagraph/tests/translators/test_nodes.py                      64      0   100%
    metagraph/tests/translators/test_vector.py                     16      0   100%
    metagraph/tests/types/__init__.py                               0      0   100%
    metagraph/tests/types/test_dataframe.py                        10      0   100%
    metagraph/tests/types/test_edges.py                           107      0   100%
    metagraph/tests/types/test_matrix.py                           41      0   100%
    metagraph/tests/types/test_nodes.py                            34      0   100%
    metagraph/tests/types/test_vector.py                           31      0   100%
    metagraph/tests/util.py                                       107      4    96%   50-51, 82, 96
    metagraph/types.py                                             34      2    94%   7, 10
    metagraph/wrappers.py                                          14      0   100%
    -----------------------------------------------------------------------------------------
    TOTAL                                                        3525    229    94%

    ========================================================== 84 passed, 1 warning in 5.67s ==========================================================

.. raw:: html

   </details>