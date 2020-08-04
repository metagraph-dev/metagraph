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
- Allow an abstract algorithm to have a default implementation that only calls
  other abstract algorithms via the current ``Resolver``.

Type System and Dispatcher
--------------------------
- Nested namespace of properties and validators, that allow common subgroups to be used in multiple types.
- Change ``unambigous_subcomponents`` to ``allowed_translations``
    - Include which property values can translate to another values
    - Clarify what self-translators need to do
- Allow concrete algorithms to specialize on more than the abstract type signature, such as literal values of
  non-translatable (i.e. "Python") types.
