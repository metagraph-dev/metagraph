.. _translators:

Translators
===========

Translators convert from one data object into another.

Here is an example translator

.. code-block:: python

    from metagraph import translator

    @translator
    def list_to_set(x: List, **props) -> Set:
        return set(x)

The signature must be properly typed as this is how Metagraph knows the input and output type
of the translator. The additional ``props`` are properties that the user wants to set in the
output.

Translators are called from the resolver, passing in the input object and specifying the desired
output type.

.. code-block:: python

    g = NetworkXGraph(...)
    g2 = resolver.translate(g, CuGraph)

The primary rule of translators is that translation can only happen
between objects of the same abstract type. There are exceptions to this rule,
but this is the main usage of translators in Metagraph.

Using the Resolver in a Translator
----------------------------------

If a translator ever needs access to the `Resolver` object which called it,
set the `include_resolver` flag in the decorator and include a "resolver" keyword argument
in the signature.

.. code-block:: python

    @translator(include_resolver=True)
    def my_translator(x: NumpyNodeMap, *, resolver, **props) -> MyCustomNodeMap:
        # resolver is now available

Unambiguous Subcomponents
-------------------------

Besides ``props``, translators cannot include additional parameters in their signature.
As a result, translators usually only convert between objects with the same abstract type.
Their underlying structure changes, but the data the object represents is unchanged.

The exception to this rule is when an abstract type specifies ``unambiguous_subcomponents``.
These are a set of other abstract types which can be safely extracted from an object.
For example, a NodeMap can be translated to a NodeSet because every NodeMap contains a
NodeSet. It's merely the set of nodes without any values. Because it doesn't require any additional
parameters and is unambiguous in meaning, the use of a translator to convert is acceptable.

Converting with arguments
-------------------------

Anything else which converts between two objects but takes additional arguments must be
written as an algorithm. For example, converting a Vector into a Matrix by reshaping requires
a (rows, cols) tuple. This would need to be written as an algorithm rather than a translator.
