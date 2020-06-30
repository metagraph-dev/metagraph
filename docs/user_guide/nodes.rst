Nodes
=====

Nodes have integer IDs.

In the simplest case, nodes are labeled 0..N-1 for a set of **N** nodes.
If a filter is performed, however, the set of nodes will no longer be sequential.

Alternatively, when loading a dataset, the serialized data may already have node ids which are
not sequential.

Data objects which handle nodes must have a mechanism for keeping track of non-sequential node ids.
This allows for translation between different types with a guarantee that the meaning of a node
id will remain consistent.

This consistency applies even when creating new data objects as a result of running an algorithm.
For example, running pagerank on an EdgeMap will return a NodeMap with node ids that remain
consistent based on the original EdgeMap.

Node Labels
-----------

Often, users want to refer to nodes by something other than an integer node ID. To help with this,
Metagraph has a ``NodeLabels`` class which provides a bidirectional mapping from node id to label.

The label may be any hashable Python object. Often this will be a string, but it is not limited to strings.

Because it is a bidirectional mapping, the number of unique labels must equal the number of unique
node ids.

A node label mapping may be created in two ways.

.. code-block:: python

    # Option 1
    nl = mg.NodeLabels([0, 2, 3], ["Sally", "Bob", "Alice"])

    # Option 2
    nl2 = mg.NodeLabels.from_dict({"Sally": 0, "Alice": 3, "Bob": 2})

Forward lookups using the label is performed using bracket notation.

.. code-block:: python

    >>> nl["Sally"]
    0
    >>> nl["Alice"]
    3
    >>> "Bob" in nl
    True

Backward looks are done using the ``.ids`` attribute.

.. code-block:: python

    >>> nl.ids[0]
    "Sally"
    >>> nl.ids[3]
    "Alice"
    >>> 1 in nl.ids
    False

To support multiple lookup, a list of labels or ids can be passed in. This is especially useful
for resolving an edge, which is simply a start and end node.

.. code-block:: python

    >>> nl[["Sally", "Bob"]]
    [0, 2]
    >>> nl.ids[[3, 2]]
    ["Alice", "Bob"]

The user is responsible for converting labels into ids when calling algorithms, and converting back
into labels when the algorithm returns ids.

In the future, Metagraph may incorporate NodeLabels into the types directly, but for now the responsibility
remains with the user.
