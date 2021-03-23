"""
Containers which mimic `typing` containers, but which allow for instances rather than only types

ex. typing.Optional[MyAbstractType] works, but typing.Optional[MyAbstractType(some_prop=True)] fails
"""
from .plugin import AbstractType, ConcreteType, MetaWrapper


# Use in signatures when a node ID is required
class NodeID:
    def __repr__(self):
        return "NodeID"

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "Do not attempt to create a NodeID. Simply pass in the node_id as an int"
        )


# Create a singleton object which masks the class
NodeID = NodeID()


class Combo:
    def __init__(self, types, *, kind=None, optional=False):
        """
        types are the list of acceptable types
        optional indicates whether None is allowed
        kind must be one of 'Union' or 'List' or None
          - Union: Allows `types` to have several values.
                   Translation is restricted to within the same abstract type family.
                   Input data must be a single object.
          - List: Restricts `types` to a single value.
                  There are no translation restrictions.
                  Input data must be a list of items of the single type.
          - None: Restricts `types` to a single value.
                  There are no translation restrictions.
                  Input data must be a single object.

        The reason for kind=Union is to allow Union[PythonNodeSetType, PythonNodeMapType].
            In this case, the algorithm can utilize either one, presumably by assuming
            the NodeSet weights are all equal to 1. However, we would not want a NumpyNodeMap
            to be translated to a PythonNodeSet, losing its weights in the process.
            To avoid this kind of mistake where valid translators exist, kind=Union
            enforces no translation across abstract type boundaries.
        """
        if kind not in {"List", "Union", None}:
            raise TypeError(f"Invalid kind: {kind}")

        if not hasattr(types, "__len__"):
            raise TypeError(
                f"Expected a list of types for kind={kind}, but got {type(types)}"
            )

        if len(types) == 0:
            raise TypeError(f"Found an empty list of types for kind={kind}")

        if kind in {"List", None} and len(types) > 1:
            raise TypeError(
                f"Expected exactly one type for kind={kind}, found {len(types)}"
            )

        if kind is None and not optional:
            raise TypeError("Combo must have a kind or be optional")

        subtype = None
        checked_types = []
        for t in types:
            if t in {None, type(None)}:
                raise TypeError(
                    "Do not include `None` in the types. Instead, set `optional=True`"
                )
            checked_types.append(t)

        self.types = checked_types
        self.optional = optional
        self.kind = kind
        self.subtype = subtype

    def __len__(self):
        return len(self.types)

    def __repr__(self):
        if self.kind == "List":
            ret = f"mg.List[{str(self.types[0])}]"
        elif self.kind == "Union":
            ret = f"mg.Union[{','.join(str(x) for x in self.types)}]"
        else:
            ret = str(self.types[0])
        if self.optional:
            ret = f"mg.Optional[{ret}]"
        return ret

    # This allows Combo to behave as a SignatureModifier
    def update_annotation(self, obj, *, name=None, index=None):
        assert name is not None
        assert index is not None
        self.types[index] = obj

    def compute_common_subtype(self):
        # Ensure all AbstractTypes or all ConcreteType or all Python types, but not mixed
        subtype = None
        for t in self.types:
            if type(t) is type:
                this_subtype = "python"
            elif isinstance(t, AbstractType):
                this_subtype = "abstract"
            elif isinstance(t, ConcreteType):
                this_subtype = "concrete"
            elif t is NodeID:
                this_subtype = "node_id"
            else:
                raise TypeError(f"Unexpected subtype within Combo: {type(t)}")

            if subtype is None:
                subtype = this_subtype
            elif subtype != this_subtype:
                raise TypeError(
                    f"Cannot mix {subtype} and {this_subtype} types within {self.kind}"
                )
        self.subtype = subtype


#################################################
# These will be converted to Singleton instances
#################################################


class List:
    """
    Similar to typing.List, except allows for instances of metagraph types
    """

    def __getitem__(self, element_type):
        if type(element_type) is not tuple:
            element_type = (element_type,)

        return Combo(element_type, kind="List")


class Union:
    """
    Similar to typing.Union, except allows for instances of metagraph types
    """

    def __getitem__(self, parameters):
        if type(parameters) is not tuple or len(parameters) < 2:
            raise TypeError(f"Union requires more than one parameter")

        kind = "Union"
        optional = False

        # Filter out None-like values
        params_filtered = []
        for p in parameters:
            if p in (None, type(None)):
                optional = True
            else:
                params_filtered.append(p)

        if optional and len(params_filtered) == 1:
            kind = None

        return Combo(params_filtered, kind=kind, optional=optional)


class Optional:
    """
    Similar to typing.Optional, except allows for instances of metagraph types
    """

    def __getitem__(self, parameter):
        if type(parameter) is tuple:
            if len(parameter) > 1:
                raise TypeError("Too many parameters, only one allowed for Optional")
            parameter = parameter[0]

        if isinstance(parameter, Combo):
            return Combo(parameter.types, kind=parameter.kind, optional=True)

        return Combo([parameter], optional=True)


# Convert to singletons
List = List()
Union = Union()
Optional = Optional()
