.. _types:

Types
=====

Types are the fundamental building block of Metagraph. They allow dispatching based on
types defined in algorithm signatures.

Types also have properties which can affect dispatching.

Types vs. Objects
-----------------

Types are not the same as data objects. Data objects hold the actual data -- node IDs, weights,
edges, etc. Types describe what the data objects mean and contain methods to compute properties
about the data objects.

For example, a Python ``list`` is a data object. It can be thought of as a Vector, but the ``list`` has
no method to report its dtype property. Vectors must have a dtype. To work around this limitation,
a ``PythonVectorType`` can be created which knows how to compute a dtype given a Python ``list``.

In this example, the Python ``list`` knows nothing about the ``PythonVectorType`` class, while the
``PythonVectorType`` **does** know about the Python ``list``.


Abstract Types
--------------

Abstract types describe a generic kind of data container with potentially many equivalent representations.

Abstract types can define allowable properties which allow for specialized versions of the type.

This is an example for the abstract type EdgeMap which describes a set of edges and their
associated data, similar to a Graph, but more limited. An EdgeMap has a dtype and can be directed or undirected.
It may or may not have negative weights.

.. code-block:: python

    class EdgeMap(AbstractType):
        properties = {
            "is_directed": [True, False],
            "dtype": DTYPE_CHOICES,
            "has_negative_weights": [True, False, None],
        }
        unambiguous_subcomponents = {EdgeSet}

An abstract type can also define ``unambiguous_subcomponents``, which is a ``set`` of
other abstract types which this type is allowed to be translated into.

Concrete Types
--------------

Concrete types describe a specific data object which fits under the abstract type category.
Many such representations can exist, but all must represent identical data, allowing for
translation between data objects of the same abstract type.

Concrete types can define properties which only apply to the specific data object.
These are specified in the ``allowed_props`` attribute. It must be a ``dict`` similar
to abstract properties.

This is a mock example of what a ScipyMatrix type might look like. It would subclass ``ConcreteType``
and indicate which abstract class it belongs to (in this case ``Matrix``).

.. code-block:: python

    class ScipyMatrixType(ConcreteType, abstract=Matrix):
        value_type = scipy.sparse.spmatrix

        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: List[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"dtype"} - ret.keys():
                if prop == "dtype":
                    ret[prop] = dtypes.dtypes_simplified[obj.dtype]

            return ret

        @classmethod
        def assert_equal(cls, obj1, obj2, aprops1, aprops2, cprops1, cprops2, *, rel_tol=1e-9, abs_tol=0.0):
            assert obj1.shape == obj2.shape, f"{obj1.shape} != {obj2.shape}"
            assert aprops1 == aprops2, f"abstract property mismatch: {props1} != {props2}"
            # additional assertions ...

In the example above, there is a ``value_type`` attribute pointing to the data object --
``scipy.sparse.spmatrix``. This is the most common form for a concrete type, pointing
to exactly one data class.

If more than one data class can be used with a concrete type, ``value_type`` is not provided
and instead the author must override ``is_typeclass_of`` so the system can properly figure out
which concrete type to use for every data object.

If any abstract properties are defined for the associated abstract type, ``_compute_abstract_properties``
must be written to compute those properties for a given object.

Concrete properties are defined in the ``allowed_props`` attribute. If this is specified,
``_compute_concrete_properties`` must be written to compute those properties for a given object.

Finally, it is recommended to write the ``assert_equal`` method for comparing two data objects
of this type. Doing so allows these objects to be used in testing.

.. _wrappers:

Wrappers
--------

Often, the data object by itself does not contain enough information to be fully understood
by Metagraph. A wrapper is needed around the data object to contain additional information.
This wrapper will still need a separate concrete type which describes it.

To aid plugin authors, a standard pattern exists to create wrappers. A wrapper must subclass
``Wrapper`` and indicate the abstract type it belongs to. It should have its own constructor
and otherwise add methods and attributes as necessary to satisfy the concept of the abstract
type.

Within the wrapper class definition, an inner class named ``TypeMixin`` must be written.
This inner class is created exactly like ``ConcreteType`` except for the following:

- It does not subclass ``ConcreteType``
- It does not define the abstract class (that is done in the Wrapper definition)
- It does not define ``value_type``

All other parts of ``ConcreteType`` are defined within the inner ``TypeMixin`` class:

- allowed_props
- _compute_abstract_properties
- _compute_concrete_properties
- assert_equal
- etc.

When the wrapper is registered with Metagraph, this ``TypeMixin`` class will be converted into
a proper ``ConcreteType`` and set as the ``.Type`` attribute on the wrapper. The ``value_type``
will point to the wrapper class, linking the two objects.

Wrapper Convenience Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Several common resolver methods are made available as shortcuts on wrappers.

- ``.translate(dst)`` will translate to another type
- ``.run(algo_name, *args, **kwargs)`` will run an algorithm using the wrapper as the first argument

This example shows equivalent calls:

.. code-block:: python

    y = mg.translate(x, "NetworkXGraph")
    y = x.translate("NetworkXGraph")

    pr = mg.algos.centrality.pagerank(x, damping=0.75)
    pr = x.run("centrality.pagerank", damping=0.75)
