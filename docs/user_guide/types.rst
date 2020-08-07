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

This is an example for the abstract type Matrix which describes a two-dimensional
block of data. A Matrix has a dtype and can be dense or not (i.e. sparse). It can be
square or not.

.. code-block:: python

    class Matrix(AbstractType):
        properties = {
            "is_dense": [False, True],
            "is_square": [False, True],
            "dtype": ["str", "float", "int", "bool"],
        }

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

This is an example of the ScipyMatrix type. It subclasses ``ConcreteType`` and indicates
which abstract class it belongs to (in this case ``Matrix``).

.. code-block:: python

    class ScipyMatrixType(ConcreteType, abstract=Matrix):
        value_type = scipy.sparse.spmatrix

        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: List[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"is_dense", "dtype", "is_square"} - ret.keys():
                if prop == "is_dense":
                    ret[prop] = False
                if prop == "dtype":
                    ret[prop] = dtypes.dtypes_simplified[obj.dtype]
                if prop == "is_square":
                    nrows, ncols = obj.shape
                    ret[prop] = nrows == ncols

            # slow properties, only compute if asked
            for prop in props - ret.keys():
                if prop == "is_symmetric":
                    ret[prop] = ret["is_square"] and (obj.T != obj).nnz == 0

            return ret

        @classmethod
        def assert_equal(cls, obj1, obj2, props1, props2, *, rel_tol=1e-9, abs_tol=0.0):
            assert obj1.shape == obj2.shape, f"{obj1.shape} != {obj2.shape}"
            assert props1 == props2, f"property mismatch: {props1} != {props2}"
            # additional assertions ...

In the example above, there is a ``value_type`` attribute pointing to the data object --
``scipy.sparse.spmatrix``. This is the most common form for a concrete type, pointing
to exactly one data class.

If more than one data class can be used with a concrete type, ``value_type`` is not provided
and instead the author must override ``is_typeclass_of`` so the system can properly figure out
which concrete type to use for every data object.

If any abstract properties were defined, ``_compute_abstract_properties`` must be written to
compute those properties for a given object.

Concrete properties are defined in the ``allowed_props`` attribute. If this is specified,
``_compute_concrete_properties`` must be written to compute those properties for a given object.

Finally, it is recommended to write the ``assert_equal`` method for comparing two data objects
of this type. Doing so allows these objects to be used in testing.

.. _wrappers:

Wrappers
--------

Often, the data object by itself does not contain enough information to be fully understood
by Metagraph. A wrapper is needed around the data object to contain additional information.
This wrapper will still need a separate Type which describes it.

To aid plugin authors, a standard pattern exists to create wrappers. A wrapper must subclass
``Wrapper`` and indicate the abstract type it belongs to. It should have its own constructor
and otherwise add methods and attributes as necessary to satisfy the concept of the abstract
type.

Within the wrapper class definition, an inner class named ``TypeMixin`` must be written.
This inner class is created exactly like ``ConcreteType`` except for the following:

- It does not subclass ``ConcreteType``
- It does not define the abstract class (that is done in the Wrapper definition)
- It does not define ``value_type``

All other parts of ``ConcreteType`` *are* defined within the inner ``TypeMixin`` class:

- allowed_props
- _compute_abstract_properties
- _compute_concrete_properties
- assert_equal
- etc.

When the wrapper is registered with Metagraph, this ``TypeMixin`` class will be converted into
a proper ``ConcreteType`` and set as the ``.Type`` attribute on the wrapper. The ``value_type``
will be set pointing to the wrapper class, linking the two objects.
