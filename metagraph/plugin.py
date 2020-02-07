"""Base classes for basic metagraph plugins.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AbstractType:
    """Equivalence class of concrete types."""

    name: str


@dataclass(frozen=True)
class ConcreteType:
    """A specific data type in a particular memory space recognized by metagraph"""

    abstract: str
    name: str
    props: dict = field(default_factory=lambda: {})

    # populated once the resolver has linked everything
    abstract_instance: AbstractType = None


@dataclass(frozen=True)
class Translator:
    """A converter from one concrete type to another"""

    srctype: str
    dsttype: str

    # populated once the resolver has linked everything
    srctype_instance: ConcreteType = None
    dsttype_instance: ConcreteType = None

    def translate(self, src, **props):
        """Return value `src` of `srctype` converted to `dsttype`"""
        raise NotImplementedError("required method")
