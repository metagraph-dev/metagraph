Style Guide
===========

Prerequisite Reading & Misc. Background Information
---------------------------------------------------

Familiarity with the concepts covered in the following sections are highly recommended:

* :ref:`Getting Started<getting_started>`
* :ref:`User Guide<user_guide>`
* :ref:`Plugin Parts<plugin_parts>`

We highly recommend becoming an expert Metagraph user before contributing to Metagraph itself.

Additionally, we highly recommend reading our `GitHub repository's README <https://github.com/ContinuumIO/metagraph>`_.

Code Formatting
---------------

We use `Black <https://black.readthedocs.io/en/stable/>`_ to format our code.

We use `pre-commit <https://pre-commit.com/>`_ to have a pre-commit Git hook that runs `Black <https://black.readthedocs.io/en/stable/>`_ to format our code prior to committing. You can do the same by running ``pre-commit install`` at the top-level of your Metagraph checkout. 

Environment Setup
-----------------

We maintain our dependencies via `conda <https://docs.conda.io/en/latest/>`_.

If your proposed changes require any new dependencies, it's necessary to update the ``environment.yml`` accordingly to avoid dependency issues for users and fellow developers.

String Formatting
-----------------

We conventionally use f-string formatting rather than string formatting via ``‘%’`` or ``‘.format()’``.

Core Plugin Style
-----------------

These are style guidelines for how to contribute to the core plugins.

Each core plugin is stored under its own plugin directory under ``metagraph/plugins/``.

3-Part Modules
~~~~~~~~~~~~~~

The :ref:`Plugin Author Guide<plugin_author_guide>` describes 3 main plugin parts, i.e. algorithms, translators, and types.

We conventionally compartmentalize these into their own modules under the plugin directory as ``algorithms.py``, ``translators.py``, and ``types.py``.

We conventionally make them available by importing them in the ``__init__.py`` of the plugin directory.

