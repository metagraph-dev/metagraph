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

.. raw:: html

   <details>
   <summary>Results from running the test suite</summary>

.. code-block::

    >>> pytest
    ============================================================================ test session starts =============================================================================
    platform darwin -- Python 3.7.7, pytest-5.4.2, py-1.9.0, pluggy-0.13.1
    rootdir: /projects/metagraph, inifile: setup.cfg, testpaths: metagraph/tests
    plugins: cov-2.8.1
    collected 197 items

    metagraph/tests/test_config.py .                                                                                                                                       [  0%]
    metagraph/tests/test_dask.py .....                                                                                                                                     [  3%]
    metagraph/tests/test_dtypes.py .                                                                                                                                       [  3%]
    metagraph/tests/test_entrypoints.py ...                                                                                                                                [  5%]
    metagraph/tests/test_explorer.py ..........                                                                                                                            [ 10%]
    metagraph/tests/test_multiverify.py .........                                                                                                                          [ 14%]
    metagraph/tests/test_node_labels.py ...                                                                                                                                [ 16%]
    metagraph/tests/test_plugin.py .........                                                                                                                               [ 20%]
    metagraph/tests/test_registry.py ..                                                                                                                                    [ 21%]
    metagraph/tests/test_resolver.py .............................                                                                                                         [ 36%]
    metagraph/tests/test_toplevel.py ...                                                                                                                                   [ 38%]
    metagraph/tests/test_typecache.py .                                                                                                                                    [ 38%]
    metagraph/tests/test_types.py ...                                                                                                                                      [ 40%]
    metagraph/tests/algorithms/test_bipartite.py .                                                                                                                         [ 40%]
    metagraph/tests/algorithms/test_centrality.py .........                                                                                                                [ 45%]
    metagraph/tests/algorithms/test_clustering.py .......                                                                                                                  [ 48%]
    metagraph/tests/algorithms/test_embedding.py ssssss.                                                                                                                   [ 52%]
    metagraph/tests/algorithms/test_flow.py ..                                                                                                                             [ 53%]
    metagraph/tests/algorithms/test_subgraph.py ....s.....                                                                                                                 [ 58%]
    metagraph/tests/algorithms/test_traversal.py ..........                                                                                                                [ 63%]
    metagraph/tests/algorithms/test_utility.py ..............s                                                                                                             [ 71%]
    metagraph/tests/translators/test_bipartite.py ..                                                                                                                       [ 72%]
    metagraph/tests/translators/test_dataframe.py .                                                                                                                        [ 72%]
    metagraph/tests/translators/test_edgemap.py .......                                                                                                                    [ 76%]
    metagraph/tests/translators/test_edgeset.py ...                                                                                                                        [ 77%]
    metagraph/tests/translators/test_graph.py ................                                                                                                             [ 85%]
    metagraph/tests/translators/test_matrix.py ...                                                                                                                         [ 87%]
    metagraph/tests/translators/test_node_map.py .....                                                                                                                     [ 89%]
    metagraph/tests/translators/test_node_set.py ....                                                                                                                      [ 91%]
    metagraph/tests/translators/test_vector.py ..                                                                                                                          [ 92%]
    metagraph/tests/types/test_dataframe.py .                                                                                                                              [ 93%]
    metagraph/tests/types/test_edges.py ...                                                                                                                                [ 94%]
    metagraph/tests/types/test_graph.py ...                                                                                                                                [ 96%]
    metagraph/tests/types/test_matrix.py ..                                                                                                                                [ 97%]
    metagraph/tests/types/test_nodes.py ...                                                                                                                                [ 98%]
    metagraph/tests/types/test_vector.py ..                                                                                                                                [100%]

    ============================================================================== warnings summary ==============================================================================
    /Users/jkitchen/miniconda3/envs/mg/lib/python3.7/site-packages/donfig/config_obj.py:30
      /Users/jkitchen/miniconda3/envs/mg/lib/python3.7/site-packages/donfig/config_obj.py:30: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3,and in 3.9 it will stop working
        from collections import Mapping

    -- Docs: https://docs.pytest.org/en/latest/warnings.html

    ---------- coverage: platform darwin, python 3.7.7-final-0 -----------
    Name                                            Stmts   Miss  Cover   Missing
    -----------------------------------------------------------------------------
    metagraph/__init__.py                              40      0   100%
    metagraph/algorithms/__init__.py                    1      0   100%
    metagraph/algorithms/bipartite.py                   4      0   100%
    metagraph/algorithms/centrality.py                 18      0   100%
    metagraph/algorithms/clustering.py                 17      0   100%
    metagraph/algorithms/embedding.py                  20      0   100%
    metagraph/algorithms/flow.py                        8      0   100%
    metagraph/algorithms/subgraph.py                   21      0   100%
    metagraph/algorithms/traversal.py                  21      0   100%
    metagraph/algorithms/utility.py                    34      0   100%
    metagraph/core/__init__.py                          0      0   100%
    metagraph/core/dask/__init__.py                     0      0   100%
    metagraph/core/dask/placeholder.py                 75     16    79%   24-25, 31, 35, 44-49, 79, 89-94
    metagraph/core/dask/resolver.py                   154     30    81%   93-95, 118-120, 161, 184, 189-208, 227, 235, 248, 262-263, 267-268
    metagraph/core/dtypes.py                           26      0   100%
    metagraph/core/entrypoints.py                      21      0   100%
    metagraph/core/multiverify.py                     192      6    97%   13-14, 20, 281, 340, 343
    metagraph/core/node_labels.py                      49      7    86%   46, 48, 58, 60, 65, 72, 76
    metagraph/core/planning.py                        271     47    83%   76, 82-86, 91-92, 96-97, 101-105, 124, 198, 215, 232-233, 293, 300, 315, 338-357, 363, 375-378, 384, 392, 397-401, 415-416, 421-425, 428
    metagraph/core/plugin.py                          290     24    92%   36, 55, 125, 129, 133, 140, 167, 208, 215, 233, 241, 286, 299, 324, 341, 390, 411, 444, 447, 475, 510, 512, 612, 616
    metagraph/core/plugin_registry.py                  68      0   100%
    metagraph/core/resolver.py                        557     54    90%   73-79, 166-174, 261, 266, 318, 380, 384, 395, 432, 461, 528, 535-539, 592, 628-638, 641, 654, 656, 754, 829-830, 854-855, 896, 904-918, 957-958, 962-963
    metagraph/core/roundtrip.py                       115      5    96%   123, 130, 206, 215, 222
    metagraph/core/typecache.py                        82      8    90%   19, 22-25, 85, 89, 100
    metagraph/core/typing.py                           71      9    87%   41, 60, 63-66, 76, 108, 113
    metagraph/dask.py                                   4      0   100%
    metagraph/explorer/__init__.py                      1      0   100%
    metagraph/explorer/api.py                         245     40    84%   27, 36, 48, 146, 206, 283-289, 293-298, 309-310, 334, 336, 338, 360-369, 382, 397-398, 425-436
    metagraph/explorer/service.py                     109    109     0%   1-169
    metagraph/plugins/__init__.py                      35      2    94%   39-42
    metagraph/plugins/graphblas/__init__.py             1      0   100%
    metagraph/plugins/graphblas/algorithms.py         100      4    96%   126-129
    metagraph/plugins/graphblas/translators.py         94      2    98%   136, 153
    metagraph/plugins/graphblas/types.py              223     30    87%   80, 83, 99-108, 117, 123, 259-274, 309, 315, 344-345, 472-474, 508-509, 520
    metagraph/plugins/networkx/__init__.py              1      0   100%
    metagraph/plugins/networkx/algorithms.py          346     25    93%   410, 418-419, 424-426, 440, 449-450, 455-456, 472, 481-482, 487-489, 518, 522, 529-530, 535-536, 555, 560
    metagraph/plugins/networkx/translators.py          27      0   100%
    metagraph/plugins/networkx/types.py               211     27    87%   12, 38, 184, 190, 196, 224-226, 241-243, 312-322, 325-335
    metagraph/plugins/numpy/__init__.py                 1      0   100%
    metagraph/plugins/numpy/algorithms.py              47      0   100%
    metagraph/plugins/numpy/translators.py             41      0   100%
    metagraph/plugins/numpy/types.py                  153     27    82%   58, 60, 66-69, 87-88, 91, 94-98, 136, 138, 146, 150, 159, 181-182, 185-189, 195-198
    metagraph/plugins/pandas/__init__.py                1      0   100%
    metagraph/plugins/pandas/algorithms.py             12      0   100%
    metagraph/plugins/pandas/translators.py            36      0   100%
    metagraph/plugins/pandas/types.py                 132     10    92%   34-35, 69, 75-76, 79, 186-187, 190, 226
    metagraph/plugins/python/__init__.py                1      0   100%
    metagraph/plugins/python/algorithms.py             30      0   100%
    metagraph/plugins/python/translators.py            22      0   100%
    metagraph/plugins/python/types.py                  40      1    98%   49
    metagraph/plugins/scipy/__init__.py                 1      0   100%
    metagraph/plugins/scipy/algorithms.py             186      3    98%   75-76, 86
    metagraph/plugins/scipy/translators.py             98      1    99%   103
    metagraph/plugins/scipy/types.py                  182      8    96%   28, 40, 118-121, 125, 232-233, 347
    metagraph/tests/__init__.py                         0      0   100%
    metagraph/tests/__main__.py                         7      7     0%   1-10
    metagraph/tests/algorithms/__init__.py              1      0   100%
    metagraph/tests/algorithms/test_bipartite.py       15      0   100%
    metagraph/tests/algorithms/test_centrality.py      80      0   100%
    metagraph/tests/algorithms/test_clustering.py      85      0   100%
    metagraph/tests/algorithms/test_embedding.py      279    196    30%   34-49, 74-109, 147-169, 195-262, 318-432, 441-513
    metagraph/tests/algorithms/test_flow.py            42      0   100%
    metagraph/tests/algorithms/test_subgraph.py       131      2    98%   103, 174
    metagraph/tests/algorithms/test_traversal.py      136      0   100%
    metagraph/tests/algorithms/test_utility.py        168      0   100%
    metagraph/tests/bad_site_dir/__init__.py            0      0   100%
    metagraph/tests/bad_site_dir2/__init__.py           0      0   100%
    metagraph/tests/conftest.py                         4      0   100%
    metagraph/tests/plugins/__init__.py                 0      0   100%
    metagraph/tests/site_dir/__init__.py                0      0   100%
    metagraph/tests/site_dir/plugin1.py                14      0   100%
    metagraph/tests/test_config.py                      7      0   100%
    metagraph/tests/test_dask.py                       81      0   100%
    metagraph/tests/test_dtypes.py                      8      0   100%
    metagraph/tests/test_entrypoints.py                20      0   100%
    metagraph/tests/test_explorer.py                   64      0   100%
    metagraph/tests/test_multiverify.py               121      0   100%
    metagraph/tests/test_node_labels.py                36      0   100%
    metagraph/tests/test_plugin.py                    106      0   100%
    metagraph/tests/test_registry.py                   32      0   100%
    metagraph/tests/test_resolver.py                  548      0   100%
    metagraph/tests/test_toplevel.py                   21      0   100%
    metagraph/tests/test_typecache.py                  28      0   100%
    metagraph/tests/test_types.py                      20      0   100%
    metagraph/tests/translators/__init__.py             1      0   100%
    metagraph/tests/translators/test_bipartite.py      29      0   100%
    metagraph/tests/translators/test_dataframe.py       8      0   100%
    metagraph/tests/translators/test_edgemap.py        43      0   100%
    metagraph/tests/translators/test_edgeset.py        16      0   100%
    metagraph/tests/translators/test_graph.py         201      0   100%
    metagraph/tests/translators/test_matrix.py         28      0   100%
    metagraph/tests/translators/test_node_map.py       47      0   100%
    metagraph/tests/translators/test_node_set.py       31      0   100%
    metagraph/tests/translators/test_vector.py         22      0   100%
    metagraph/tests/types/__init__.py                   0      0   100%
    metagraph/tests/types/test_dataframe.py            10      0   100%
    metagraph/tests/types/test_edges.py                74      0   100%
    metagraph/tests/types/test_graph.py               103      0   100%
    metagraph/tests/types/test_matrix.py               19      0   100%
    metagraph/tests/types/test_nodes.py                33      0   100%
    metagraph/tests/types/test_vector.py               19      0   100%
    metagraph/tests/util.py                           110      6    95%   56-57, 103, 107, 109-110
    metagraph/types.py                                 42      0   100%
    metagraph/wrappers.py                              17      0   100%
    -----------------------------------------------------------------------------
    TOTAL                                            7442    706    91%

    ================================================================= 189 passed, 8 skipped, 1 warning in 15.94s =================================================================

.. raw:: html

   </details>