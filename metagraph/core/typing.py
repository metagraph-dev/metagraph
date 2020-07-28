"""
Containers which mimic `typing` containers, but which allow for instances rather than only types

ex. typing.Optional[MyAbstractType] works, but typing.Optional[MyAbstractType(some_prop=True)] fails
"""
from .plugin import AbstractType, ConcreteType, MetaWrapper


class Combo:
    def __init__(self, types, *, optional=False):
        # Ensure all AbstractTypes or all ConcreteType or all Python types, but not mixed
        kind = None
        checked_types = []
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
            else:
                raise TypeError(f"type within Union or Optional may not be {type(t)}")

            if kind is None:
                kind = this_kind
            elif kind != this_kind:
                raise TypeError(f"Cannot mix {kind} and {this_kind} types within Union")

            checked_types.append(t)

        self.types = checked_types
        self.optional = optional
        self.kind = kind

    def __len__(self):
        return len(self.types)


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
            return Combo(parameter.types, optional=True)

        return Combo([parameter], optional=True)


# Convert to singleton
Optional = Optional()
