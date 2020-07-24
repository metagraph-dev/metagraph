Roadmap
=======

This page captures plans and ideas for improving Metagraph.  These items are
not in any particular order, and some items on this list may not happen.  Feel
free to open issues in the `issue tracker
<https://github.com/ContinuumIO/metagraph/issues>`_ if you are interested in
discussing the details of one of these ideas.


Project Structure
-----------------

- Split packaging of metagraph into a minimal core package with abstract
  types and algorithms, and put the basic plugins into separate packages. The
  ``metagraph`` package then becomes a metapackage that installs core and a
  generally useful set of default plugins.
- The multi-dispatch and automatic translation system in Metagraph is not
  graph-specific and could be split out into a separate package.

General Features
----------------

- Lazy execution of algorithms by returning a Dask task graph from algorithm
  calls. 
- Global preference setting plugin priority.  This would be used to
  more easily switch a workflow from one backend to another without having to
  modify the calculation itself.

Type System
-----------
- [insert list of type system enhancements here]

Algorithms
----------
- [insert list of future algorithms here]