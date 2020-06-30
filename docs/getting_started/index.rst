.. _getting_started:

Getting Started
===============

.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   tutorial
   concepts
   use_case_1_airline_connectedness
   use_case_2_netflix_kevin_bacon


Metagraph is a plugin-based system for performing graph computations.

Rather than being a set of components on which graph algorithms are built,
it serves as an orchestration layer on top of existing graph libraries.

Key components include:
  - A friendly user-centric API of :ref:`types` and :ref:`algorithms`
  - A plugin system where existing graph libraries can be combined and reused
  - A lazy-execution scheduler which choose between several equivalent implementations of each algorithm
