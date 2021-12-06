Data Loading and Export
=======================

Generally, users load data into a type that Metagraph is aware of, and has a
translation path to the desired graph type via the plugins that have been
installed.  Metagraph also provides special support for loading data into a 
distributed graph structure (see `Distributed COO to CSR conversion`_).

Loading
-------

For example, to load an edge list from a CSV file, a user might use pandas and build a DataFrame.

.. code-block:: python

    import pandas as pd
    df = pd.read_csv(csv_file)

To be used by Metagraph, this needs to be a known data object. Pandas DataFrames are known in Metagraph,
but they are interpreted as a DataFrame. If we wanted this edge list to be interpreted as an EdgeSet, we
would need to construct it ourselves.

.. code-block:: python

    es = mg.wrappers.EdgeSet.PandasEdgeSet(df, src_label="Column1", dst_label="Column2")

Now ``es`` can be used within Metagraph and passed to algorithms which expect an EdgeSet. Translation
and all the other Metagraph magic will work.

Exporting
---------

When the final result is computed and a user wants to retrieve the results from the data object,
again the process is mostly manual for the user.

The first step is to get the data in the right data format. Calling ``translate`` on the resolver is the
easiest way to convert to the desired type.

If the data object is a Wrapper object, the actual data object will be need to be retrieved. The typical
convention for this is to have a ``.value`` attribute holding on to the actual data object. This is only a
convention and may not be honored by all plugins. Additionally, some wrappers hold multiple data objects
and there is no convention for secondary data object attribute names. Looking at the wrapper's docstring
or inspecting the code is the preferred way to understand how the internal pieces are stored.

Mutating Objects
----------------

In general, wrappers in Metagraph assume a non-mutating data object as input. If a wrapper is constructed
from an object which is then mutated, the wrapper will almost certainly see the mutation. This is problematic
because Metagraph caches properties which may not be valid after the mutation.

Even without using a wrapper, Metagraph will cache the properties of raw data objects used in Metagraph
translations and algorithms. Mutating the raw data object may cause unexpected behavior because the
reported properties may be invalid.

In short, do not mutate objects which are still in active use within Metagraph. There is a way of forcing
the property cache to clear if mutation is impossible to avoid, but this is generally discouraged.


Distributed COO to CSR conversion
---------------------------------

Metagraph has a special interface for loading COO graph data in a Dask
Dataframe into a distributed CSR graph data structure.  Metagraph does not
provide any distributed graph data structures itself, but a Metagraph plugin
may offer a CSR graph object that is distributed across the Dask worker
systems and is therefore visible to all workers.  In this situation, Metagraph
offers support for parallel loading of data into the distributed CSR object.

.. autofunction:: metagraph.core.dask.loader::load_coo_to_csr

Because of the variety of potential distributed CSR implementations,
``load_coo_to_csr`` requires a loader class that implements all of the following methods:

.. autoclass:: metagraph.core.dask.loader::CSRLoader
    :members:

For details on how to create a new loader (especially how to manage shared
resource lifetimes), see the implementation of the ``SharedCSRLoader`` class.
