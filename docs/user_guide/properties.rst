.. _properties:

Properties
==========

Properties allow for the specialization of types and are used to indicate requirements
of the data in an algorithm signature.

For example, an EdgeMap has an ``is_directed`` property which can be ``True`` or ``False``.
Strongly connected components can indicate that it takes ``EdgeMap(is_directed=True)`` in its
signature. Metagraph will ensure that strongly connected components is only called with data
that satisfies this requirement.

Properties can apply to abstract types or to concrete types. If they apply to abstract types,
they are referred to as abstract properties. Likewise, concrete properties apply to concrete types.

Abstract properties deal with aspects of a type which are common across all implementations.
EdgeMap with ``is_directed`` is an example of this. All implementations of an EdgeMap must handle
directed and undirected variants.

Concrete properties deal with implementation-specific details and are more rare than abstract properties.


Specification
-------------

Properties are defined when a Type is defined.

Here is an example of the abstract Matrix type with its properties:

.. code-block:: python

    class Matrix(AbstractType):
        properties = {
            "is_dense": [False, True],
            "is_square": [False, True],
            "dtype": ["str", "float", "int", "bool"],
        }

The ``properties`` attribute must be a ``dict`` with string keys and list-like keys.
Generally, the values should be a list of True/False or a list of strings.

The keys represent the properties for this type. The values represent the allowable values
for each property. Together, they define the specification of a type's properties.

Requirements
------------

Within an algorithm's signature, properties can be passed to a type class
to indicate what assumptions the algorithm will make. Metagraph will enforce
these restrictions prior to calling the algorithm.

Here is an example of strongly connected components. Besides annotating the type
of ``graph`` as ``EdgeSet``, this also indicates that the ``is_directed`` property
of the EdgeSet must be ``True``.

.. code-block:: python

    @abstract_algorithm("clustering.strongly_connected_components")
    def strongly_connected_components(graph: EdgeSet(is_directed=True)) -> NodeMap:
        pass

If a list of values is allowable, this is indicated with a list or set of values in the signature.
Here is the pagerank signature showing the two allowable dtypes are ``{"int", "float"}``.

.. code-block:: python

    @abstract_algorithm("link_analysis.pagerank")
    def pagerank(
        graph: EdgeMap(dtype={"int", "float"}),
        damping: float = 0.85,
        maxiter: int = 50,
        tolerance: float = 1e-05,
    ) -> NodeMap:
        pass


Classification
--------------

Concrete types are required to have the following functions:
  - .. py:function:: _compute_abstract_properties(obj, props: List[str], known_props: Dict[str, Any]) -> Dict[str, Any]
  - .. py:function:: _compute_concrete_properties(obj, props: List[str], known_props: Dict[str, Any]) -> Dict[str, Any]

These are used to discover the properties of data objects and ensure they meet the
algorithm requirements.

If an algorithm signature leaves out a property, that property value is not passed to the
``_compute_xxx_properties`` method, effectively ignoring it.

If an algorithm signature specifies a value or values, the computed property must match
or the object will be rejected prior to the algorithm call.

Property Cache
~~~~~~~~~~~~~~

Computing properties of an object might be expensive. To avoid duplicating this effort each time the object
is passed to an algorithm, properties are cached by metagraph.

When requesting new properties to be computed, the ``known_props`` are passed along to avoid
redundant work.

Translation functions can populate the property cache by calling the concrete type's ``get_typeinfo``
method and then calling ``update_props`` on the TypeInfo object. This helps avoid unnecessary effort
to compute properties that the translator already knows from the input object.

If for some reason, the property cache for an object needs to be cleared, this is the way to do it.
In general, this should not be needed for normal usage of Metagraph.

.. code-block:: python

    # How to force cached properties to be purged
    SomeConcreteType._typecache.expire(obj)
