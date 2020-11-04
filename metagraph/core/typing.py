"""
Containers which mimic `typing` containers, but which allow for instances rather than only types

ex. typing.Optional[MyAbstractType] works, but typing.Optional[MyAbstractType(some_prop=True)] fails
"""
from .plugin import AbstractType, ConcreteType, MetaWrapper
from ..types import NodeID


class Combo:
    def __init__(self, types, *, optional=False, strict=None):
        # Ensure all AbstractTypes or all ConcreteType or all Python types or all UniformIterable, but not mixed
        kind = None
        checked_types = set()
        for t in types:
            if t is None or t is type(None):
                optional = True
                continue

            # Convert all AbstractTypes and ConcreteTypes into instances
            if type(t) is type and issubclass(t, (AbstractType, ConcreteType)):
                t = t()

            # Convert all Wrappers into instances of their Type class
            if type(t) is MetaWrapper:
                t = t.Type()

            if type(t) is type:
                this_kind = "python"
            elif isinstance(t, AbstractType):
                this_kind = "abstract"
            elif isinstance(t, ConcreteType):
                this_kind = "concrete"
            elif t is NodeID:
                this_kind = "node_id"
            elif isinstance(t, UniformIterable):
                this_kind = "uniform_iterable"
            elif getattr(t, "__origin__", None) in {list}:  # expand this as necessary
                this_kind = "uniform_iterable"
            else:
                raise TypeError(f"type within Union or Optional may not be {type(t)}")

            if kind is None:
                kind = this_kind
            elif kind != this_kind:
                raise TypeError(f"Cannot mix {kind} and {this_kind} types within Union")

            checked_types.add(t)

        if strict is None:
            # Assume a single type with optional=True is only meant to be optional, not strict
            strict = False if len(checked_types) == 1 and optional else True

        self.types = checked_types
        self.optional = optional
        self.kind = kind
        self.strict = strict

    def __len__(self):
        return len(self.types)

    def __repr__(self):
        ret = f"Union[{','.join(str(x) for x in self.types)}]"
        if self.optional:
            ret = f"Optional[{ret}]"
        return ret


class Union:
    """
    Similar to typing.Union, except allows for instances of metagraph types
    """

    def __getitem__(self, parameters):
        if len(parameters) < 2:
            raise TypeError(f"Expected more than one parameter, got {len(parameters)}")

        return Combo(parameters, optional=False)


# Convert to singleton
Union = Union()


class Optional:
    """
    Similar to typing.Optional, except allows for instances of metagraph types
    """

    def __getitem__(self, parameter):
        if isinstance(parameter, Combo):
            return Combo(parameter.types, optional=True, strict=parameter.strict)

        return Combo([parameter], optional=True, strict=False)


# Convert to singleton
Optional = Optional()


class UniformIterable:
    def __init__(self, element_type, container_name):
        if type(element_type) is type and issubclass(
            element_type, (AbstractType, ConcreteType)
        ):
            element_type = element_type()
        if type(element_type) is MetaWrapper:
            element_type = element_type.Type()
        self.element_type = element_type
        self.container_name = container_name

    def __repr__(self):
        return f"{self.container_name}[{self.element_type}]"


class List:
    def __getitem__(self, element_type):
        return UniformIterable(element_type, self.__class__.__qualname__)


# Convert to singleton
List = List()
