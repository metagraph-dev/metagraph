"""Base classes for basic metagraph plugins.
"""
import inspect


class AbstractType:
    """Equivalence class of concrete types."""

    # all instances of an abstract type are equivalent!
    def __eq__(self, other):
        return self.__class__ == other.__class__

    def __hash__(self):
        return hash(self.__class__)


class ConcreteType:
    """A specific data type in a particular memory space recognized by metagraph"""

    # Most subclasses only need to set these class attributes
    abstract = None  # must override this
    value_class = None  # override this for fast path type identification
    allowed_props = {}  # default is no props
    target = "cpu"  # key may be used in future to guide dispatch

    # Override these methods only if necessary
    def __init__(self, **props):
        for key in props:
            if key not in self.allowed_props:
                raise KeyError(f"{key} not allowed property of {self.__class__}")
            # maybe type check?
        self.props = dict(props)

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
        if isinstance(obj, cls.value_class):
            return cls()  # no properties to specialize on
        else:
            raise TypeError(f"object not of type {cls.__class__}")


class Translator:
    def __init__(self, func):
        self.func = func
        self.src_type = None
        self.dst_type = None

    def __call__(self, src, **props):
        return self.func(src, **props)


# decorator
def translator(func):
    # FIXME: signature checks?
    return Translator(func)


def normalize_type(t):
    if issubclass(t, ConcreteType):
        return t()
    else:
        return t


def normalize_parameter(p):
    return p.replace(annotation=normalize_type(p.annotation))


def normalize_signature(sig):
    """Return normalized signature with bare type classes instantiated"""
    new_params = [normalize_parameter(p) for p in sig.parameters.values()]
    new_return = normalize_type(sig.return_annotation)
    return sig.replace(parameters=new_params, return_annotation=new_return)


class AbstractAlgorithm:
    def __init__(self, func, name):
        self.func = func
        self.name = name

    def get_signature(self):
        return inspect.signature(self.func)


def abstract_algorithm(name):
    def _abstract_decorator(func):
        return AbstractAlgorithm(func=func, name=name)

    return _abstract_decorator


class ConcreteAlgorithm:
    def __init__(self, func, abstract_name):
        self.func = func
        self.abstract_name = abstract_name

    def get_signature(self):
        return normalize_signature(inspect.signature(self.func))

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def concrete_algorithm(abstract_name):
    def _concrete_decorator(func):
        return ConcreteAlgorithm(func=func, abstract_name=abstract_name)

    return _concrete_decorator
