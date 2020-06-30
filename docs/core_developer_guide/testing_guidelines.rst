Testing Guidelines
==================

Prerequisite Reading & Misc. Background Information
---------------------------------------------------

Familiarity with the concepts covered in the following sections are highly recommended:

* :ref:`Getting Started<getting_started>`
* :ref:`User Guide<user_guide>`
* :ref:`Plugin Parts<plugin_parts>`

Testing Utilities
-----------------

We use `pytest <https://docs.pytest.org/en/stable/>`_ and `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_.

Tests are stored in *metagraph/tests/*.

Core Plugin Testing
~~~~~~~~~~~~~~~~~~~

Be sure to read :ref:`our guidelines on plugin testing<plugin_testing>` and the :ref:`Plugin Author Guide<plugin_author_guide>`.

We store tests for the 3 plugin parts described in the :ref:`Plugin Author Guide<plugin_author_guide>` in 3
subdirectories of *metagraph/tests/*.

Tests for core plugin algorithms are stored in *metagraph/tests/algorithms/*.

Tests for core plugin translators are stored in *metagraph/tests/translators/*.

Tests for core plugin types are stored in *metagraph/tests/types/*. 

Core metagraph Testing
~~~~~~~~~~~~~~~~~~~~~~

Tests for metagraph's core functionality, e.g. resolver, plugin registry, etc. are stored directly in *metagraph/tests/*.

When making core metagraph changes, ensure via `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ that test
coverage is as close to 100% as possible.
