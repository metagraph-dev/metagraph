"""A cache for type information on objects that does not mutate the object and
tracks object lifetime to remove records as needed.
"""

from typing import Dict, List, Iterable, Any
import weakref
from dataclasses import dataclass


@dataclass
class TypeInfo:
    abstract_typeclass: Any
    known_abstract_props: Dict[str, Any]
    concrete_typeclass: Any
    known_concrete_props: Dict[str, Any]

    @property
    def known_props(self):
        return {**self.known_abstract_props, **self.known_concrete_props}

    def update_props(self, other):
        if type(other) is not TypeInfo:
            raise TypeError(f"other must be TypeInfo, not {type(other)}")
        self.known_abstract_props.update(other.known_abstract_props)
        self.known_concrete_props.update(other.known_concrete_props)


class TypeCache:
    """Maintains a cache of type information and properties for objects.

    Objects are not modified.  Instead this cache object maintains a weak
    reference to the object, using its id() as a key.  When the object is
    removed, the record will automatically deleted.

    Note that the cache cannot determine if the class has mutated in a way
    that invalidates the cached properties.  It is up to the user of this
    class to deal with that situation, in which case the expire() method can
    be called to manually remove the cached properties.
    """

    def __init__(self):
        self._cache = {}

    def __getitem__(self, obj):
        key = self._key(obj)
        return self._cache[key]

    def __setitem__(self, obj, typeinfo):
        key = self._key(obj)
        self._cache[key] = typeinfo
        # clean up automatically if we can
        try:
            weakref.finalize(obj, self._expire_key, key)
        except TypeError:
            # some built-in types are not weakref-able
            pass

    def __delitem__(self, obj):
        key = self._key(obj)
        del self._cache[key]

    def __contains__(self, obj):
        key = self._key(obj)
        return key in self._cache

    def __len__(self):
        return len(self._cache)

    def expire(self, obj):
        """Like del, but quietly proceed if obj typeinfo hasn't been cached yet."""
        key = self._key(obj)
        self._expire_key(key)

    def _key(self, obj):
        try:
            return hash(obj)
        except TypeError:
            return id(obj)

    def _expire_key(self, key):
        if key in self._cache:
            del self._cache[key]
