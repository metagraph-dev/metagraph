.. _end_to_end_plugin_pathway:

End-to-End Plugin Pathway
=========================

Prerequisite Reading
--------------------

Familiarity with the concepts covered in the following sections are highly recommended:

* :ref:`User Guide<user_guide>`
* :ref:`Plugin Parts<plugin_parts>`

Introduction
------------

This document provides a recommended workflow aimed at helping plugin authors write their first Metagraph plugin.

Write The Plugin
----------------

The first step is to implement the desired functionality a plugin is intended to provide. 

It's recommended to create Python modules for the :ref:`types, algorithms, translators, etc.<plugin_parts>` comprising a plugin.

Make Plugin Findable by Metagraph
---------------------------------

Once a plugin is implemented, we must make Metagraph aware of it.

Let's assume that our plugin is implemented as a module named *my_module.py*:

 .. code-block:: python
		 
		 from metagraph.plugins.networkx.types import NetworkXEdgeMap
		 from metagraph.plugins.pandas.types import PandasEdgeMap
		 import networkx as nx
		 
		 @translator
		 def edgemap_from_pandas(x: PandasEdgeMap, **props) -> NetworkXEdgeMap:
		     cur_props = PandasEdgeMap.Type.compute_abstract_properties(x, ["is_directed"])
		     if cur_props["is_directed"]:
		         out = nx.DiGraph()
		     else:
		         out = nx.Graph()			 
		     g = x.value[[x.src_label, x.dst_label, x.weight_label]]
		     out.add_weighted_edges_from(g.itertuples(index=False, name="WeightedEdge"))
		     return NetworkXEdgeMap(out, weight_label="weight",)

For the sake of simplicity, our plugin implemented by *my_module.py* only contains one translator (named *edgemap_from_pandas*) that translates a Pandas edge map to a NetworkX edge map.

This code below will make the plugin findable by Metagraph (we'll go into the exact details futher below).

*registry.py*

 .. code-block:: python

		 from metagraph import PluginRegistry
		 registry = PluginRegistry("my_plugin")
		 def find_plugins():
		     from . import my_module
		     registry.register_from_modules(my_module)
		     return registry.plugins

*setup.py*

 .. code-block:: python

		 from setuptools import setup
		 setup(
		     name="my-plugin",
		     entry_points={
		         "metagraph.plugins": "plugins=registry:find_plugins"
		     },
		 )

We'll now go over what happens in the above code.

To make the plugin findable by Metagraph, we must make an entrypoint under the name "metagraph.plugins" for a plugin-finder function (in this example *find_plugins*) that returns the plugins. For a more detailed explanation of how to use `entry points via setuptools <https://setuptools.readthedocs.io/en/latest/setuptools.html>`_, we recommend starting off with `this tutorial <https://amir.rachum.com/blog/2017/07/28/python-entry-points/>`_.

A plugin-finder function takes no inputs and returns a dictionary describing all the plugins found. We’ll go over the specifics what this dictionary looks like later in this tutorial. For now, we'll show how to use a plugin registry to generate this dictionary. 

As shown in the *registry.py* example above, a plugin registry can import all the relevant plugins from given modules via the *register_from_modules* method (note how we registered ``my_module`` in ``find_plugins``). This method imports all the translators, concrete algorithms, etc. from the modules (which are often easily recognized via the use of the decorators shown in :ref:`Plugin Parts<plugin_parts>`).

A plugin registry is initialized with a default plugin name ("my_plugin" in the *registry.py* example above).

*register_from_modules* has a keyword parameter of *name* that denotes the plugin name to attach the registered abstract types, wrappers, etc. to. If *name* is not specified, the default plugin name is used.

Using a plugin registry has the following properties:

* The plugin registry raises exceptions for plugin name conflicts.
* The plugin registry raises exceptions for duplicate registration of the same concrete types, abstract algorithms, etc.
* The plugin registry raises exceptions when concrete algorithm signatures don't match abstract algorithm signatures. 
* The plugin registry automatically searches modules passed to *register_from_modules* for wrappers, translators, etc. This allows for separation of plugin functionality into different Python modules.

A plugin registry doesn't actually inform Metagraph of anything. It is simply a datastructure that registers and sanity checks plugins.

Plugin registries have a ``plugins`` attribute that is a dictionary describing all the plugins known to the plugin registry. Plugin-finder functions should return dictionaries like this since this is what will be used by Metagraph to account for the found plugins.

An entrypoint declaration (e.g. as is shown in our *setup.py* example above) pointing to the plugin-finder function is what informs Metagraph of the plugins.

Labelling Abstract Types, Translators, Concrete Algorithms, etc.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We'll now go over how to label translators, wrappers, concrete types, etc. in such a way that plugin registries can take note of them (when the *register_from_modules* is invoked) to create these dictionaries.

There are decorators for abstract algorithms, concrete algorithms, and translators that will make them findable by the *register_from_modules* method. These decorators are:

* ``abstract_algorithm``
* ``concrete_algorithm``
* ``translator``

To make abstract types, concrete types, and wrappers findable by the the *register_from_modules* method, we must define them as being subclasses of certain Metagraph base classes. 

Abstract types must be subclasses of ``metagraph.AbstractType``.

Concrete types must be subclasses of ``metagraph.ConcreteType``.

Wrappers must be subclasses of ``metagraph.Wrapper``.

Using the above decorators and class along with a plugin registry is preferred when writing a plugin-finder function as it reduces complexity.

Fine Details of Plugin Finder Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We'll now go over the specifics of what the returned dictionary from a plugin-finder function looks like.

In our toy ``my_plugin`` example, it might look like this:

 .. code-block:: python

		 {
		     'my_plugin':
		         {
			     'translators': {<metagraph.core.plugin.Translator object at 0x123456789>},
			 },
		 }

In this example, ``<metagraph.core.plugin.Translator object at 0x123456789>`` refers to our ``edgemap_from_pandas`` translator from earlier. The ``translator`` decorator creates a translator object (to be used by Metagraph) from our ``edgemap_from_pandas`` function.

In a situation where more plugins are found, the dictionary might look like this:

 .. code-block:: python

		 {
		     'plugin_a':
		         {
			     'abstract_types': {abstract_type_a_1, abstract_type_a_2, ...},
			     'abstract_algorithms': {abstract_algorithm_a_1, abstract_algorithm_a_2, ...},
			     'concrete_types': {concrete_type_a_1, concrete_type_a_2, ...},
			     'concrete_algorithms': {concrete_algorithm_a_1, concrete_algorithm_a_2, ...},
			     'wrappers': {wrapper_a_1, wrapper_a_2, ...},
			     'translators': {translator_a_1, translator_a_2, ...},
			 },
		     'plugin_b':
		         {
			     'abstract_types': {abstract_type_b_1, abstract_type_b_2, ...},
			     'abstract_algorithms': {abstract_algorithm_b_1, abstract_algorithm_b_2, ...},
			     'concrete_types': {concrete_type_b_1, concrete_type_b_2, ...},
			     'concrete_algorithms': {concrete_algorithm_b_1, concrete_algorithm_b_2, ...},
			     'wrappers': {wrapper_b_1, wrapper_b_2, ...},
			     'translators': {translator_b_1, translator_b_2, ...},
			 },
		     'plugin_c':
		         {
			     'concrete_types': {},
			     'concrete_algorithms': {concrete_algorithm_c_1, concrete_algorithm_c_2, ...},
			     'wrappers': {wrapper_c_1, wrapper_c_2, ...},
			     'translators': {translator_c_1, translator_c_2, ...},
			 },
		    ...
		 }

The keys are plugin names.

The values are dictionaries describing the plugin. 

Valid keys of a dictionary describing a plugin are:

* :ref:`'abstract_types'<types>`
* :ref:`'abstract_algorithms'<algorithms>`
* :ref:`'concrete_types'<types>`
* :ref:`'concrete_algorithms'<algorithms>`
* :ref:`'wrappers'<wrappers>`
* :ref:`'translators'<translators>`

The values of a dictionary describing a plugin are sets of values corresponding to the key, e.g. the values for the key 'translators' is a set of :ref:`translators<translators>`.

For small plugins, it's possible to explicitly create this dictionary returned by the plugin-finder function.

For larger plugins, this is difficult to maintain. Thus, explicitly creating such dictionaries is highly discouraged.

We highly encourage using a plugin registries for this reason.
