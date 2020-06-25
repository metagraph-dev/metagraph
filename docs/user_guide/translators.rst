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

The signature must be properly typed, as that is how Metagraph knows the input and output type
of the translator. The additional ``props`` are properties that the user wants set in the output.

Translators are called from the resolver, passing in the input object and specifying the desired
output type.

.. code-block:: python

    g = NetworkXGraph(...)
    g2 = resolver.translate(g, CuGraph)


Unambiguous Subcomponents
-------------------------

Besides ``props``, translators cannot include additional parameters in their signature.
As a result, translators usually only convert between objects with the same abstract type.
Their underlying structure changes, but the data the object represents is unchanged.

The exception to this rule is when an abstract type specifies ``unambiguous_subcomponents``.
These are a set of other abstract types which can be safely extracted from an object.
For example, a NodeMap can be translated to a NodeSet because it doesn't require any additional
parameters and is unambiguous in meaning.

Converting with arguments
-------------------------

Anything else which converts between two objects but takes additional arguments must be
written as an algorithm. For example, converting a Vector into a Matrix by reshaping requires
a (rows, cols) tuple. This would need to be written as an algorithm rather than a translator.