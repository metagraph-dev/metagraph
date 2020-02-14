"""Base classes for basic metagraph plugins.
"""
import types
import inspect
from typing import Callable


class AbstractType:
    """Equivalence class of concrete types."""

    # all instances of an abstract type are equivalent!
    def __eq__(self, other):
        return self.__class__ == other.__class__

    def __hash__(self):
        return hash(self.__class__)

    def __init_subclass__(cls, *, registry=None):
        # Usually types are decorated with @registry.register,
        # but this provides another valid way
        if registry is not None:
            registry.register(cls)


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
        """Set required properties of for this type with keyword arguments"""
        for key in props:
            if key not in self.allowed_props:
                raise KeyError(f"{key} not allowed property of {self.__class__}")
            # maybe type check?
        self.props = dict(props)

    def __init_subclass__(cls, *, abstract=None, registry=None):
        """Enforce requirements on 'abstract' attribute"""
        super().__init_subclass__()

        if abstract is None:
            raise TypeError(f"Missing required 'abstract' keyword argument on {cls}.")
        elif not isinstance(abstract, type) or not issubclass(abstract, AbstractType):
            raise TypeError(
                f"'abstract' keyword argument on {cls} must be subclass of AbstractType"
            )
        cls.abstract = abstract

        # Usually types are decorated with @registry.register,
        # but this provides another valid way
        if registry is not None:
            registry.register(cls)

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
        assert len(cls.allowed_props) == 0  # must override if there are properties
        if isinstance(obj, cls.value_type):
            return cls()  # no properties to specialize on
        else:
            raise TypeError(f"object not of type {cls.__class__}")


class Wrapper:
    """Helper class for creating wrappers around data objects

    A ConcreteType will be automatically created with its `value_type` set to this class.
    The auto-created ConcreteType will be attached as `.Type` onto the wrapper class.
    """

    # These class attributes will be passed on to the created ConcreteType
    allowed_props = {}  # default is no props
    target = "cpu"  # key may be used in future to guide dispatch

    def __init_subclass__(cls, *, abstract=None, registry=None):
        cls.Type = types.new_class(
            f"{cls.__name__}Type", (ConcreteType,), {"abstract": abstract}
        )
        cls.Type.__module__ = cls.__module__
        cls.Type.value_type = cls
        cls.Type.allowed_props = cls.allowed_props
        cls.Type.target = cls.target

        # Usually wrappers are decorated with @registry.register,
        # but this provides another valid way
        if registry is not None:
            registry.register(cls)


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


# decorator
def translator(func: Callable, *, registry=None):
    # FIXME: signature checks?
    trans = Translator(func)
    if registry is not None:
        registry.register(trans)
    return trans


def normalize_type(t):
    """Instantiate ConcreteType classes with no properties (found in signatures)"""
    if issubclass(t, ConcreteType):
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


def abstract_algorithm(name: str, *, registry=None):
    def _abstract_decorator(func: Callable):
        algo = AbstractAlgorithm(func=func, name=name)
        if registry is not None:
            registry.register(algo)
        return algo

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


def concrete_algorithm(abstract_name: str, *, registry=None):
    def _concrete_decorator(func: Callable):
        algo = ConcreteAlgorithm(func=func, abstract_name=abstract_name)
        if registry is not None:
            registry.register(algo)
        return algo

    return _concrete_decorator
