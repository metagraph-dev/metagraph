import types
import dask
from dask import is_dask_collection
from dask.base import DaskMethodsMixin, tokenize
from dask.core import quote, flatten
from dask.highlevelgraph import HighLevelGraph
from metagraph.core.plugin import ConcreteAlgorithm, Translator, ConcreteType
from metagraph.core.compiler import optimize
from .visualize import visualize
from .tasks import DelayedAlgo, DelayedTranslate


def single_key(seq):
    return seq[0]


def _taskify(arg, dsk):
    if is_dask_collection(arg):
        arg, graph = finalize(arg)
        dsk.update(graph)
    else:
        arg = quote(arg)
    return arg


def taskify(arg, dsk):
    if isinstance(arg, (tuple, list, set)):
        typ = type(arg)
        return typ(_taskify(a, dsk) for a in arg)
    else:
        return _taskify(arg, dsk)


def rebuild(dsk, cls, key):
    return cls(key, dsk)


def ph_apply(func, args, kwargs):
    return func(*args, **kwargs)


def finalize(collection):
    assert is_dask_collection(collection)

    if isinstance(collection, Placeholder):
        return collection._key, collection._dsk

    name = "finalize-" + tokenize(collection)
    keys = collection.__dask_keys__()
    finalize, args = collection.__dask_postcompute__()
    layer = {name: (finalize, keys) + args}
    graph = HighLevelGraph.from_collections(name, layer, dependencies=[collection])
    return name, graph


class Placeholder(DaskMethodsMixin):
    """
    Acts as a stand-in for the actual `value_type` of a ConcreteType in a delayed context
    """

    concrete_type = None  # subclasses should override this
    __dask_scheduler__ = staticmethod(dask.threaded.get)

    def __init__(self, key, dsk=None):
        self._key = key
        if dsk is None:  # pragma: no cover
            dsk = {}
        self._dsk = dsk

    @property
    def key(self):
        return self._key

    def __dask_graph__(self):
        return self._dsk

    def __dask_keys__(self):
        return [self._key]

    def __dask_tokenize__(self):
        return self._key

    def __dask_postcompute__(self):
        return single_key, ()

    def __dask_postpersist__(self):
        return rebuild, (self.__class__, self._key)

    @staticmethod
    def __dask_optimize__(dsk, keys, **kwargs):
        return optimize(dsk, output_keys=list(flatten(keys)), **kwargs)

    def visualize(self, filename="mydask", format=None, optimize_graph=False, **kwargs):
        return visualize(
            self,
            filename=filename,
            format=format,
            optimize_graph=optimize_graph,
            **kwargs,
        )

    @classmethod
    def build(
        cls,
        key,
        func,
        args,
        kwargs=None,
        source_type=None,
        result_type=None,
        resolver=None,
    ):
        dsk = {}
        new_args = []
        for arg in args:
            arg = taskify(arg, dsk)
            new_args.append(arg)

        new_kwargs_flat = []
        if kwargs is None:
            kwargs = {}
        for kw, val in kwargs.items():
            val = taskify(val, dsk)
            new_kwargs_flat.append([kw, val])
        # Add this func to the task graph
        if isinstance(func, ConcreteAlgorithm):
            task_func = DelayedAlgo(func, result_type=result_type)
            dsk[key] = (task_func, new_args, (dict, new_kwargs_flat))
        elif isinstance(func, Translator):
            task_func = DelayedTranslate(
                func,
                source_type=source_type,
                result_type=result_type,
                resolver=resolver,
            )
            dsk[key] = (task_func, new_args, (dict, new_kwargs_flat))
        else:
            task_func = ph_apply
            dsk[key] = (task_func, func, new_args, (dict, new_kwargs_flat))

        return cls(key, dsk)


class DelayedWrapper:
    """
    Wraps a class constructor and returns the indicated Placeholder when called.
    This allows for delayed construction of objects which know the eventual type,
    enabling translations and algorithm calls using the delayed object as input.
    """

    def __init__(self, klass, placeholder):
        self._klass = klass
        self._ph = placeholder

    def __call__(self, *args, **kwargs):
        key = (
            f"init-{tokenize(self._ph, self._klass, args, kwargs)}",
            self._klass.__name__,
        )
        return self._ph.build(key, self._klass, args, kwargs)

    def __repr__(self):
        return f"DelayedWrapper<{self._ph.concrete_type.__name__}>"

    def __getattr__(self, item):
        return getattr(self._klass, item)
