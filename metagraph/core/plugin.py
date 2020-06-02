"""Base classes for basic metagraph plugins.
"""
import types
import inspect
from typing import Callable, List, Dict, Any


class AbstractType:
    """Equivalence class of concrete types."""

    # Properties must be a list of values from most general to most narrow
    properties = {}

    def __init_subclass__(cls, **kwargs):
        # Check properties are lists
        for key, val in cls.properties.items():
            if not isinstance(val, (list, tuple)):
                raise KeyError(
                    f"{key} is an invalid property; must be of type list, not {type(val)}"
                )
            cls.properties[key] = tuple(val)

    def __init__(self, **props):
        # Start with all properties at the most general level
        prop_idx = {key: 0 for key in self.properties}
        prop_val = {key: self.properties[key][0] for key in self.properties}
        for key, val in props.items():
            if key not in self.properties:
                raise KeyError(f"{key} not a valid property of {self.__class__}")
            try:
                idx = self.properties[key].index(val)
                prop_idx[key] = idx
                prop_val[key] = val
            except ValueError:
                raise ValueError(
                    f"Invalid setting for {key} property: '{val}'; must be one of {self.properties[key]}"
                )
        self.prop_idx = prop_idx
        self.prop_val = prop_val

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.prop_idx == other.prop_idx

    def __hash__(self):
        return hash((self.__class__, tuple(self.prop_idx.items())))

    def __getitem__(self, key):
        return self.prop_val[key]


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
    abstract_property_specificity_limits = (
        {}
    )  # highest specificity supported for abstract properties
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
    def _validate_abstract_props(cls, props: List[str]) -> bool:
        for propname in props:
            if propname not in cls.abstract.properties:
                raise KeyError(
                    f"{propname} is not an abstract property of {cls.abstract.__name__}"
                )

    @classmethod
    def _validate_concrete_props(cls, props: List[str]) -> bool:
        for propname in props:
            if propname not in cls.allowed_props:
                raise KeyError(
                    f"{propname} is not an concrete property of {cls.__name__}"
                )

    @classmethod
    def compute_abstract_properties(cls, obj, props: List[str]) -> Dict[str, Any]:
        """Return a dictionary with a subset of abstract properties for this object.

        At a minimum, only the requested properties will be computed, although
        this method may return additional keys if they can be computed with
        minimal additional cost.  The return value from this method should be
        cached by the caller to avoid recomputing properties repeatedly.
        """
        if len(cls.abstract.properties) > 0:
            raise NotImplementedError(
                "Must override `compute_abstract_properties` if type has abstract properties"
            )
        if len(props) > 0:
            raise TypeError("This type has no abstract properties")
        else:
            return {}

    @classmethod
    def compute_concrete_properties(cls, obj, props: List[str]) -> Dict[str, Any]:
        """Return a dictionary with a subset of concrete properties for this object.

        At a minimum, only the requested properties will be computed, although
        this method may return additional keys if they can be computed with
        minimal additional cost.  The return value from this method should be
        cached by the caller to avoid recomputing properties repeatedly.
        """
        if len(cls.allowed_props) > 0:
            raise NotImplementedError(
                "Must override `compute_concrete_properties` if type has concrete properties"
            )
        if len(props) > 0:
            raise TypeError("This type has no concrete properties")
        else:
            return {}

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
    def assert_equal(cls, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0) -> bool:
        """
        Compare whether obj1 and obj2 are equal, raising an AssertionError if not equal.
        rel_tol and abs_tol should be used when comparing floating point numbers
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

    # These class attributes will be passed on to the created ConcreteType
    allowed_props = {}  # default is no props
    abstract_property_specificity_limits = (
        {}
    )  # highest specificity supported for abstract properties
    target = "cpu"  # key may be used in future to guide dispatch

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

        cls.Type = types.new_class(
            f"{cls.__name__}Type", (ConcreteType,), {"abstract": abstract}
        )
        cls.Type.__module__ = cls.__module__
        cls.Type.__doc__ = cls.__doc__
        # Copy objects and methods from wrapper to Type class
        cls.Type.value_type = cls
        cls.Type.allowed_props = cls.allowed_props
        cls.Type.abstract_property_specificity_limits = (
            cls.abstract_property_specificity_limits
        )
        cls.Type.target = cls.target
        for funcname in ["is_satisfied_by", "is_satisfied_by_value"]:
            if hasattr(cls, funcname):
                func = getattr(cls, funcname)
                setattr(cls.Type, funcname, func)
                delattr(cls, funcname)
        for methodname in [
            "is_typeclass_of",
            "compute_abstract_properties",
            "compute_concrete_properties",
            "get_type",
            "assert_equal",
        ]:
            if hasattr(cls, methodname):
                func = getattr(cls, methodname).__func__
                setattr(cls.Type, methodname, classmethod(func))
                delattr(cls, methodname)
        for sfuncname in []:
            if hasattr(cls, sfuncname):
                func = getattr(cls, sfuncname)
                setattr(cls.Type, sfuncname, staticmethod(func))
                delattr(cls, sfuncname)

    @staticmethod
    def _assert_instance(obj, klass, err_msg=None):
        if not isinstance(obj, klass):
            if err_msg:
                raise TypeError(err_msg)
            else:
                raise TypeError(f"{obj} is not an instance of {klass.__name__}")

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

    def __init__(self, func: Callable):
        self.func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__wrapped__ = func

    def __call__(self, src, **props):
        return self.func(src, **props)


def translator(func: Callable = None):
    """
    decorator which can be called as either:
    >>> @translator
    >>> def myfunc(): ...

    We also handle the format
    >>> @translate()
    >>> def myfunc(): ...
    """
    # FIXME: signature checks?
    if func is None:
        return Translator
    else:
        return Translator(func)


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

    def __init__(self, func: Callable, name: str):
        self.func = func
        self.name = name
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__wrapped__ = func
        self.__signature__ = inspect.signature(self.func)


def abstract_algorithm(name: str):
    def _abstract_decorator(func: Callable):
        return AbstractAlgorithm(func=func, name=name)

    return _abstract_decorator


class ConcreteAlgorithm:
    """A specific implementation of an abstract algorithm.

    Function signature should consist of ConcreteTypes that are compatible
    with the AbstractTypes in the corresponding abstract algorithm.  Python
    types (which are not converted) must match exactly.
    """

    def __init__(self, func: Callable, abstract_name: str):
        self.func = func
        self.abstract_name = abstract_name
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__wrapped__ = func
        self.__signature__ = normalize_signature(inspect.signature(self.func))

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def concrete_algorithm(abstract_name: str):
    def _concrete_decorator(func: Callable):
        return ConcreteAlgorithm(func=func, abstract_name=abstract_name)

    return _concrete_decorator
