"""Base classes for basic metagraph plugins.
"""
import types
import inspect
from functools import partial
from typing import Callable, List, Dict, Set, Union, Any
from .typecache import TypeCache, TypeInfo


class AbstractType:
    """Equivalence class of concrete types."""

    # Properties must be a dict of property name to set of allowable values
    # A value of None indicates unspecified value
    properties = {}

    # Unambiguous subcomponents is a set of other abstract types which can be
    # extracted without any additional information, allowing translators to be
    # written from this type to the listed subcomponents
    unambiguous_subcomponents = set()

    def __init_subclass__(cls, **kwargs):
        # Check properties are lists
        for key, val in cls.properties.items():
            if not isinstance(val, set):
                cls.properties[key] = set(val)

    def __init__(self, **props):
        prop_val = {key: None for key in self.properties}
        for key, val in props.items():
            if key not in self.properties:
                raise KeyError(f"{key} not a valid property of {self.__class__}")
            if isinstance(val, (set, tuple, list)):
                for v in val:
                    if v not in self.properties[key]:
                        raise ValueError(
                            f"Invalid setting for {key} property: '{v}'; must be one of {self.properties[key]}"
                        )
                prop_val[key] = tuple(sorted(val))  # sort to give consistent hash
            else:
                if val not in self.properties[key]:
                    raise ValueError(
                        f"Invalid setting for {key} property: '{val}'; must be one of {self.properties[key]}"
                    )
                prop_val[key] = val
        self.prop_val = prop_val

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.prop_val == other.prop_val

    def __hash__(self):
        return hash((self.__class__, tuple(self.prop_val.items())))

    def __getitem__(self, key):
        return self.prop_val[key]

    def __repr__(self):
        props_clean = {k: v for k, v in self.prop_val.items() if v is not None}
        return f"{self.__class__.__name__}({props_clean})"


class ConcreteType:
    """A specific data type in a particular memory space recognized by metagraph.

    Subclasses of ConcreteType pass an `abstract` keyword argument in the
    inheritance definition:

        class MyConcreteType(ConcreteType, abstract=MyAbstractType):
            pass


    For faster dispatch, set the `value_type` attribute to the Python class
    which is uniquely associated with this type.

    In type signatures, the uninstantiated class is considered equivalent to
    an instance with no properties set.
    """

    # Most subclasses only need to set these class attributes
    value_type = None  # override this for fast path type identification
    allowed_props = {}  # default is no props
    target = "cpu"  # key may be used in future to guide dispatch

    # Override these methods only if necessary
    def __init__(self, **props):
        """
        Used in two ways:
        1. As a requirements indicator
           Specify concrete properties which are required for the algorithm
        2. As a descriptor of a concrete type instance
           Includes both concrete and abstract properties which describe the instance
        """
        # Separate abstract properties from concrete properties
        abstract_keys = props.keys() & self.abstract.properties.keys()
        abstract_props = {key: props.pop(key) for key in abstract_keys}
        if abstract_props:
            self.abstract_instance = self.abstract(**abstract_props)
        else:
            self.abstract_instance = None
        # Handle concrete properties
        for key in props:
            if key not in self.allowed_props:
                raise KeyError(f"{key} not allowed property of {self.__class__}")
            # maybe type check?
        self.props = dict(props)

    def __init_subclass__(cls, *, abstract=None):
        """Enforce requirements on 'abstract' attribute"""
        super().__init_subclass__()

        if abstract is None:
            raise TypeError(f"Missing required 'abstract' keyword argument on {cls}.")
        elif not isinstance(abstract, type) or not issubclass(abstract, AbstractType):
            raise TypeError(
                f"'abstract' keyword argument on {cls} must be subclass of AbstractType"
            )
        cls.abstract = abstract
        # Property caches live with each ConcreteType, allowing them to be easily accessible
        # separate from the Resolver
        cls._typecache = TypeCache()
        # Ensure ConcreteType.method decorators are used in ConcreteType class
        # They are intended only to be used in a Wrapper class
        for name, val in cls.__dict__.items():
            if getattr(val, "_is_type_method", False):
                raise TypeError(
                    "Invalid decorator: `ConcreteType.method` should only be used in a Wrapper class"
                )
            elif getattr(val, "_is_type_classmethod", False):
                raise TypeError(
                    "Invalid decorator: `ConcreteType.classmethod` should only be used in a Wrapper class"
                )
            elif getattr(val, "_is_type_staticmethod", False):
                raise TypeError(
                    "Invalid decorator: `ConcreteType.staticmethod` should only be used in a Wrapper class"
                )

    @classmethod
    def get_typeinfo(cls, value):
        if not hasattr(cls, "_typecache"):
            raise NotImplementedError("Only implemented for subclasses of ConcreteType")

        if value in cls._typecache:
            return cls._typecache[value]

        # Add a new entry for value
        typeinfo = TypeInfo(
            abstract_typeclass=cls.abstract,
            known_abstract_props={},
            concrete_typeclass=cls,
            known_concrete_props={},
        )
        cls._typecache[value] = typeinfo
        return typeinfo

    @classmethod
    def preset_abstract_properties(cls, value, **props):
        """
        Before abstract properties are defined, they can be explicitly set using this method.
        This avoids unnecessary work when a translator knows the abstract properties.
        It can also be used to resolve ambiguous properties such as whether a symmetric matrix
            is a directed edge set or not. Normally, this would be interpreted as being undirected.
            In order to create a directed edge set with bidirectional edges, preset the property.
        """
        aprops = cls.get_typeinfo(value).known_abstract_props
        for prop, val in props.items():
            if prop in aprops:
                raise ValueError(
                    '"Cannot preset "{prop}"; already set in abstract properties'
                )
            aprops[prop] = val

    def is_satisfied_by(self, other_type):
        """Is other_type and its properties compatible with this type?

        (self must be equivalent or less specific than other_type)
        """
        if isinstance(other_type, self.__class__):
            for k in self.props:
                if k not in other_type.props or self.props[k] != other_type.props[k]:
                    return False
        else:
            return False
        return True

    def is_satisfied_by_value(self, obj):
        """Is the type associated with this object compatible with this type?

        (self must be equivalent or less specific than the type of obj)

        Note that this is potentially slow because it uses get_type() and
        therefore computes all properties.  Prefer is_satisfied_by() with a
        partially specified type instance.
        """
        try:
            t = self.get_type(obj)
            return self.is_satisfied_by(t)
        except TypeError:
            return False

    def __eq__(self, other_type):
        return isinstance(other_type, self.__class__) and self.props == other_type.props

    def __hash__(self):
        return hash((self.__class__, tuple(self.props.items())))

    def __getitem__(self, key):
        if key in self.abstract.properties:
            return self.abstract_instance[key]
        return self.props[key]

    def __repr__(self):
        if self.abstract_instance is None:
            props_clean = {}
        else:
            props_clean = {
                k: v
                for k, v in self.abstract_instance.prop_val.items()
                if v is not None
            }
        props_clean.update(
            {k: v for k, v in self.allowed_props.items() if v is not None}
        )
        return f"{self.__class__.__name__}({props_clean})"

    @classmethod
    def is_typeclass_of(cls, obj):
        """Is obj described by this type class?"""

        # check fastpath
        if cls.value_type is not None:
            return isinstance(obj, cls.value_type)
        else:
            raise NotImplementedError(
                "Must override `is_typeclass_of` if cls.value_type not set"
            )

    @classmethod
    def _compute_abstract_properties(
        cls, obj, props: Set[str], known_props: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "Must override `_compute_abstract_properties` if type has abstract properties"
        )

    @classmethod
    def compute_abstract_properties(
        cls, obj, props: Union[str, Set[str]]
    ) -> Dict[str, Any]:
        """Return a dictionary with a subset of abstract properties for this object.

        At a minimum, only the requested properties will be computed, although
        this method may return additional keys if they can be computed with
        minimal additional cost.

        The properties are cached to speed up future calls for the same properties.
        """
        assert cls.is_typeclass_of(
            obj
        ), f"Cannot compute abstract properties of {obj} using {cls}"

        if len(props) == 0:
            return {}

        # Upgrade single string to a 1-element set
        if isinstance(props, str):
            props = {props}

        # Validate properties
        for propname in props:
            if propname not in cls.abstract.properties:
                raise KeyError(
                    f"{propname} is not an abstract property of {cls.abstract.__name__}"
                )

        if type(props) is not set:
            props = set(props)

        typeinfo = cls.get_typeinfo(obj)
        abstract_props = cls._compute_abstract_properties(
            obj, props, typeinfo.known_abstract_props
        )

        # Verify requested properties were computed
        uncomputed_properties = props - set(abstract_props)
        if uncomputed_properties:
            raise AssertionError(
                f"Requested abstract properties were not computed: {uncomputed_properties}"
            )

        # Cache properties
        typeinfo.known_abstract_props.update(abstract_props)

        return abstract_props

    @classmethod
    def _compute_concrete_properties(
        cls, obj, props: Set[str], known_props: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "Must override `_compute_concrete_properties` if type has concrete properties"
        )

    @classmethod
    def compute_concrete_properties(
        cls, obj, props: Union[str, Set[str]]
    ) -> Dict[str, Any]:
        """Return a dictionary with a subset of concrete properties for this object.

        At a minimum, only the requested properties will be computed, although
        this method may return additional keys if they can be computed with
        minimal additional cost.

        The properties are cached to speed up future calls for the same properties.
        """
        assert cls.is_typeclass_of(
            obj
        ), f"Cannot compute concrete properties of {obj} using {cls}"

        if len(props) == 0:
            return {}

        # Upgrade single string to a 1-element set
        if isinstance(props, str):
            props = {props}

        # Validate properties
        for propname in props:
            if propname not in cls.allowed_props:
                raise KeyError(
                    f"{propname} is not a concrete property of {cls.__name__}"
                )

        typeinfo = cls.get_typeinfo(obj)
        concrete_props = cls._compute_concrete_properties(
            obj, props, typeinfo.known_concrete_props
        )

        # Verify requested properties were computed
        uncomputed_properties = props - set(concrete_props)
        if uncomputed_properties:
            raise AssertionError(
                f"Requested concrete properties were not computed: {uncomputed_properties}"
            )

        # Cache properties
        typeinfo.known_concrete_props.update(concrete_props)

        return concrete_props

    @classmethod
    def get_type(cls, obj):
        """Get an instance of this type class that fully describes obj

        Note that this will completely specialize the type and may require
        non-trivial computation to determine all properties of obj. Prefer to
        use is_typeclass_of(), compute_abstract_properties(), and
        compute_concrete_properties() instead of this method when possible.
        """
        if cls.is_typeclass_of(obj):
            abstract_props = cls.compute_abstract_properties(
                obj, cls.abstract.properties.keys()
            )
            concrete_props = cls.compute_concrete_properties(
                obj, cls.allowed_props.keys()
            )

            ret_val = cls(**abstract_props, **concrete_props)
            return ret_val
        else:
            raise TypeError(f"object not of type {cls.__name__}")

    @classmethod
    def assert_equal(
        cls,
        obj1,
        obj2,
        aprops1,
        aprops2,
        cprops1,
        cprops2,
        *,
        rel_tol=1e-9,
        abs_tol=0.0,
    ):
        """
        Compare whether obj1 and obj2 are equal, raising an AssertionError if not equal.
        rel_tol and abs_tol should be used when comparing floating point numbers
        props1 and props2 and dicts of all properties and can be used when performing the comparison
        """
        raise NotImplementedError()


class MetaWrapper(type):
    def __new__(mcls, name, bases, dict_, abstract=None, register=True):
        kwargs = {}
        if bases:
            kwargs["register"] = register
            if abstract is not None:
                kwargs["abstract"] = abstract
        cls = type.__new__(mcls, name, bases, dict_, **kwargs)
        if register and abstract is not None:
            # Check for required methods defined on abstract
            for name, val in abstract.__dict__.items():
                if getattr(val, "_is_required_method", False):
                    if not hasattr(cls, name):
                        raise TypeError(
                            f"{cls.__name__} is missing required wrapper method '{name}'"
                        )
                    prop = getattr(cls, name)
                    if not callable(prop):
                        raise TypeError(
                            f"{cls.__name__}.{name} must be callable, not {type(prop)}"
                        )
                if getattr(val, "_is_required_property", False):
                    if not hasattr(cls, name):
                        raise TypeError(
                            f"{cls.__name__} is missing required wrapper property '{name}'"
                        )
                    prop = getattr(cls, name)
                    if type(prop) is not property:
                        raise TypeError(
                            f"{cls.__name__}.{name} must be a property, not {type(prop)}"
                        )
        return cls


class Wrapper(metaclass=MetaWrapper):
    """Helper class for creating wrappers around data objects

    A ConcreteType will be automatically created with its `value_type` set to this class.
    The auto-created ConcreteType will be attached as `.Type` onto the wrapper class.
    """

    _resolver = None

    def __init_subclass__(cls, *, abstract=None, register=True):
        if not register:
            cls._abstract = abstract
            return

        # Attempt to lookup abstract from unregistered wrapper superclass
        implied_abstract = getattr(cls, "_abstract", None)
        if abstract is None:
            abstract = implied_abstract
        elif implied_abstract is not None:
            if abstract is not implied_abstract:
                raise TypeError(
                    f"Wrong abstract type for wrapper: {abstract}, expected {implied_abstract}"
                )

        # Use TypeMixin class to create a new ConcreteType class; store as `.Type`
        if not hasattr(cls, "TypeMixin") or type(cls.TypeMixin) is not type:
            raise TypeError(
                f"class {cls.__name__} does not define required `TypeMixin` inner class"
            )
        cls.Type = types.new_class(
            f"{cls.__name__}Type", (cls.TypeMixin, ConcreteType), {"abstract": abstract}
        )
        cls.Type.__module__ = cls.__module__
        cls.Type.__doc__ = cls.__doc__
        # Point new Type class at this wrapper
        cls.Type.value_type = cls

    def __init__(self, *, aprops=None):
        if aprops is not None:
            self.Type.preset_abstract_properties(self, **aprops)

    @staticmethod
    def _assert_instance(obj, klass, err_msg=None):
        if not isinstance(obj, klass):
            if err_msg:
                raise TypeError(err_msg)
            else:
                if type(klass) is tuple:
                    name = tuple(kls.__name__ for kls in klass)
                else:
                    name = klass.__name__
                raise TypeError(f"{obj} is not an instance of {name}")

    @staticmethod
    def _assert(cond, err_msg):
        if not cond:
            raise TypeError(err_msg)

    @staticmethod
    def required_method(func):
        func._is_required_method = True
        return func

    @staticmethod
    def required_property(func):
        func._is_required_property = True
        return func


class Translator:
    """Converts from one concrete type to another, enforcing properties on the
    destination if requested."""

    def __init__(self, func: Callable, include_resolver: bool):
        self.func = func
        self._include_resolver = include_resolver
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__wrapped__ = func

    def __call__(self, src, *, resolver=None, **props):
        if self._include_resolver:
            if resolver is None:
                raise ValueError("`resolver` is None, but is required by translator")
            if hasattr(resolver, "_resolver"):  # DaskResolver
                resolver = resolver._resolver
            return self.func(src, resolver=resolver, **props)
        else:
            return self.func(src, **props)


def translator(func: Callable = None, *, include_resolver: bool = False):
    """
    decorator which can be called as either:
    >>> @translator
    >>> def myfunc(x: FromType, **props) -> ToType: ...

    We also handle the format
    >>> @translate()
    >>> def myfunc(x: FromType, **props) -> ToType: ...

    If the resolver is needed as part of the translator, use this format
    >>> @translate(include_resolver=True)
    >>> def myfunc(x: FromType, *, resolver, **props) -> ToType: ...
    """
    # FIXME: signature checks?
    if func is None:
        return partial(Translator, include_resolver=include_resolver)
    else:
        return Translator(func, include_resolver=include_resolver)


def normalize_type(t):
    """Instantiate ConcreteType classes with no properties (found in signatures)"""
    if type(t) is type and issubclass(t, ConcreteType):
        return t()
    else:
        return t


def normalize_parameter(p: inspect.Parameter):
    """Instantiate any ConcreteType classes found in this parameter annotation"""
    return p.replace(annotation=normalize_type(p.annotation))


def normalize_signature(sig: inspect.Signature):
    """Return normalized signature with bare type classes instantiated"""
    new_params = [normalize_parameter(p) for p in sig.parameters.values()]
    new_return = normalize_type(sig.return_annotation)
    return sig.replace(parameters=new_params, return_annotation=new_return)


class AbstractAlgorithm:
    """A named algorithm with a type signature of AbstractTypes and/or Python types.

    Abstract algorithms should have empty function bodies.
    """

    def __init__(self, func: Callable, name: str, *, version: int = 0):
        self.func = func
        self.name = name
        self.version = version
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__wrapped__ = func
        self.__signature__ = inspect.signature(self.func)


def abstract_algorithm(name: str, *, version: int = 0):
    def _abstract_decorator(func: Callable):
        return AbstractAlgorithm(func=func, name=name, version=version)

    _abstract_decorator.version = version
    return _abstract_decorator


class ConcreteAlgorithm:
    """A specific implementation of an abstract algorithm.

    Function signature should consist of ConcreteTypes that are compatible
    with the AbstractTypes in the corresponding abstract algorithm.  Python
    types (which are not converted) must match exactly.
    """

    def __init__(
        self,
        func: Callable,
        abstract_name: str,
        *,
        version: int = 0,
        include_resolver: bool = False,
    ):
        self.func = func
        self.abstract_name = abstract_name
        self.version = version
        self._include_resolver = include_resolver
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__wrapped__ = func
        self.__original_signature__ = inspect.signature(self.func)
        self.__signature__ = normalize_signature(self.__original_signature__)

    def __call__(self, *args, resolver=None, **kwargs):
        if self._include_resolver:
            if resolver is None:
                raise ValueError(
                    "`resolver` is None, but is required by concrete algorithm"
                )
            if hasattr(resolver, "_resolver"):  # DaskResolver
                resolver = resolver._resolver
            return self.func(*args, resolver=resolver, **kwargs)
        else:
            return self.func(*args, **kwargs)


def concrete_algorithm(
    abstract_name: str, *, version: int = 0, include_resolver: bool = False
):
    def _concrete_decorator(func: Callable):
        return ConcreteAlgorithm(
            func=func,
            abstract_name=abstract_name,
            version=version,
            include_resolver=include_resolver,
        )

    _concrete_decorator.version = version
    return _concrete_decorator
