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
        self._fingerprints = {}

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
            key = id(obj)
            # Special handling for objects which cannot be a weakref
            try:
                weakref.ref(obj)
            except TypeError:
                if type(obj) is set:
                    self._special_handling_set(obj, key)
                elif type(obj) is dict:
                    self._special_handling_dict(obj, key)
                else:
                    raise TypeError(
                        f"Object of type {type(obj)} requires special handling which not been defined yet"
                    )
            return key

    def _expire_key(self, key):
        if key in self._cache:
            del self._cache[key]

    def _special_handling_set(self, obj, ident):
        # Nothing to do for sets until NodeSet gains an abstract property
        pass

    def _special_handling_dict(self, obj, ident):
        # Use a fingerprint to detect if this is the same object
        # - size of dict
        # - few random keys and the type and id of their values
        fingerprint_size = 3
        if ident in self._fingerprints:
            fingerprint = self._fingerprints[ident]
            try:
                assert len(obj) == fingerprint["size"]
                for key in fingerprint["keys"]:
                    assert id(obj[key]) == fingerprint[f"_id_{key}"]
                    assert type(obj[key]) == fingerprint[f"_type_{key}"]
            except (AssertionError, KeyError):
                self._expire_key(ident)
                del self._fingerprints[ident]

        if ident not in self._fingerprints:
            size = len(obj)
            if size < fingerprint_size:
                rand_keys = tuple(obj.keys())
            else:
                key_iter = iter(obj)
                rand_keys = tuple(next(key_iter) for _ in range(fingerprint_size))
            fingerprint = {"size": len(obj), "keys": rand_keys}
            for key in rand_keys:
                fingerprint[f"_id_{key}"] = id(obj[key])
                fingerprint[f"_type_{key}"] = type(obj[key])
            self._fingerprints[ident] = fingerprint
