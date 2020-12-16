.. _existing_plugins:

Community Plugins
=================

Metagraph comes with core plugins, e.g. NetworkX, SciPy, etc., but can be extended and integrated with external plugins that don't come with Metagraph by default.

Plugins are written and maintained by the community.

To use with Metagraph, simply install the plugin into your Python environment.

Metagraph uses the ``entrypoints`` Python mechanism from ``setup.py`` to automatically
find all plugins which are compatible with Metagraph. For documentation on writing a
plugin, see the :ref:`plugin author guide<plugin_author_guide>`.

Plugins we know about
---------------------

  - `metagraph-cuda <https://github.com/metagraph-dev/metagraph-cuda>`__
  - `metagraph-igraph <https://github.com/metagraph-dev/metagraph-igraph>`__
  - `metagraph-karateclub <https://github.com/metagraph-dev/metagraph-karateclub>`__
  - `metagraph-stellargraph <https://github.com/metagraph-dev/metagraph-stellargraph>`__
  - `metagraph-cogdl <https://github.com/metagraph-dev/metagraph-cogdl>`__
