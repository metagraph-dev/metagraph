"""Base classes for basic metagraph plugins.
"""
import types
import inspect
from typing import Callable


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
        for key, val in props.items():
            if key not in self.properties:
                raise KeyError(f"{key} not a valid property of {self.__class__}")
            try:
                idx = self.properties[key].index(val)
                prop_idx[key] = idx
            except ValueError:
                raise ValueError(
                    f"Invalid setting for {key} property: '{val}'; must be one of {self.properties[key]}"
                )
        self.prop_idx = prop_idx

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.prop_idx == other.prop_idx

    def __hash__(self):
        return hash((self.__class__, tuple(self.prop_idx.items())))


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
        return True

    def is_satisfied_by_value(self, obj):
        """Is the type associated with this object compatible with this type?

        (self must be equivalent or less specific than the type of obj)
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

    @classmethod
    def is_typeof(cls, obj):
        """Is obj described by this type?"""
        try:
            cls.get_type(obj)
            return True
        except TypeError:
            return False

    @classmethod
    def get_type(cls, obj):
        """Get an instance of this type class that describes obj"""
        # Must override if there are properties
        if cls.allowed_props:
            raise NotImplementedError(
                "Must override `get_type` if type has concrete properties"
            )

        if isinstance(obj, cls.value_type):
            ret_val = cls()  # no properties to specialize on
            ret_val.abstract_instance = cls.abstract()
            return ret_val
        else:
            raise TypeError(f"object not of type {cls.__name__}")


class Wrapper:
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

    def __init_subclass__(cls, *, abstract=None):
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
        for methodname in ["is_typeof", "get_type"]:
            if hasattr(cls, methodname):
                func = getattr(cls, methodname).__func__
                setattr(cls.Type, methodname, classmethod(func))
        for sfuncname in []:
            if hasattr(cls, sfuncname):
                func = getattr(cls, sfuncname)
                setattr(cls.Type, sfuncname, staticmethod(func))

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
