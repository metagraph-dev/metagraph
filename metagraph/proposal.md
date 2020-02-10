Here's a go at describing how to unify the ideas in both #2 and #3.  My assumptions are that we need to specify:

* Relations between Abstract types (`Graph`, `DenseArray`, etc) and Concrete types (`NumPyArray`, etc)
* Relations between Concrete types (`NumPyArray`) and actual data objects / classes (`numpy.ndarray`)
* Properties of Concrete types (which may also be unspecified)
* Wrappers for data that augment externally defined classes (like NetworkXGraph) with additional attributes (like `weight_name`) needed to allow the external object to be fully convertable between other concrete types.
* Type signatures for abstract functions
* Type signatures for concrete functions, which may also have concrete types that specify properties
* Collections of Abstract and concrete types / functions that will be used for dispatching and automatic translation.

There are still a bunch of aesthetic/functional choices here, so here's my proposed choices with some explanation:

* Type classes are distict from data objects:

  - This is to allow a separation of concerns between "what is this thing?" and the thing itself. (This flexibility may be needed to deal with duck typing or protocols)
  - Type classes are used in type signatures (in the way that the python `typing` module declares `List` to describe list)
  - *Instances* of type classes can be used to specify property requirements in type signatures.  Ex: `IncidenceMatrix(transposed=False)` asks for a particular property, whereas all properties not listed (as in `IncidenceMatrix` or `IncidenceMatrix()`) are implicitly "any".
  - In principle this allows us to not *have* to create a data object wrapper around every third party class, only those that are underspecified.
  - Where possible, it would be nice for metagraph to consume data from the user without imposing an additional "data preparation" step.

* For now, let's use actual the type classes in plugins, rather than strings.  If this creates some circular import issues, we can optionally allow strings to be used as substitutes, similar to the convention with other Python type signatures.

* Python inheritance is not used across the abstract and concrete type classes divide.  Inheritance may be used between abstract types classes where needed.

* Python inheritance is not used between concrete type classes and the data objects they describe.  This is because those objects may be defined in codebases we cannot modify.

* Python type signatures are used to describe both abstract and concrete algorithms for metagraph.

* Types in those signatures which are instances of the metagraph abstract or concrete types will participate in the automatic translation and dispatch mechanism.  Types which are basic python types (like `int`, `str` or `tuple`) represent "scalars" which will be passed by value without conversion.


## Worked example

Let's use `Graph` and `WeightedGraph` to see how this works.

### Abstract types

``` python
class GraphType(AbstractType):
    '''A graph is a collection of nodes and edges that connect nodes.'''
    pass  # nothing more to specify

class WeightedGraphType(Graph):
    '''A Graph that specifies a numeric weight value for each edge'''
    # a weighted graph can be converted to a graph, but a graph 
    # cannot be converted to a weighted graph
    pass
```

Note that abstract types are basically just a class and a docstring, with inheritance showing how things might be related to each other.

### Wrapper classes

An instance of `networkx.DiGraph` meets our requirement for `Graph` but not `WeightedGraph`.  For a weighted graph, we will need to define an extra wrapper class to carry the attribute name of the weight.

``` python
class NetworkXWeightedGraph:
    def __init__(self, graph, weight_label):
        self.graph = graph
        self.weight_label = weight_label
        assert isinstance(graph, nx.DiGraph)
        assert (
                weight_label in graph.nodes(data=True)[0]
        ), f"Graph is missing specified weight label: {weight_label}"
```

### Concrete types

Now we need some types to describe the NetworkX instances above.  Let's assume our base `ConcreteType` looks like this:

``` python
class ConcreteType:
    allowed_props = frozendict()  # default is no props
    target = 'cpu'  # key may be used in future to guide dispatch 

    def __init__(self, abstract, **props):
        self.abstract = abstract
        for key in props:
            if key not in allowed_props:
                raise KeyError(f'{key} not allowed property of {self.__class__}')
            # maybe type check?
        self.props = dict(props)  # copying to be paranoid

    def is_satisfied_by(self, other_type):
        # check if other_type is at least as specific as this one
        if isinstance(other_type, self.__class__):
            for k in self.props:
                if self.props[k] != other_type.props[k]:
                    return False
        return True

    def __eq__(self, other_type):
        return isinstance(other_type, self.__class__) and \
            self.props == other.props

    def isinstance(self, obj):
        # Return True if obj is an object described by this Concrete Type
        raise NotImplementedError()

    def get_props(self, obj):
        # Return a dict of properties that this object satisfies
        raise NotImplementedError()
```

The type for the NetworkX graph is:

``` python
class NetworkXGraphType(ConcreteType):
    allowed_props = frozendict({
        # placeholders for now
        'foo': bool,
        'bar': int,
    })

    def __init__(self, **props):
        super().__init__(GraphType, **props)

    def isinstance(self, obj):
        # is obj and instance of this metagraph type?
        return isinstance(obj, nx.DiGraph)
```

And the weighed graph is:
``` python
class NetworkXGraphType(ConcreteType):
    allowed_props = frozendict({
        # placeholders for now
        'baz': str,
    })

    def __init__(self, **props):
        super().__init__(WeightedGraphType, **props)

    def isinstance(self, obj):
        # is obj and instance of this metagraph type?
        return isinstance(obj, NetworkXWeightedGraph)
```

### Translator

A translator is a function that takes a value of one concrete type and maps it to a value of another concrete type (optionally with the desired type properties asserted).  A translator might look like this:

``` python
@metagraph.translator
def nx_to_cugraph(src: NetworkXGraphType, **props) -> CuGraphType:
    # insert implementation here
```

For simplicity of dispatch, a translator must be able to handle all properties of both the source and destination concrete type.  The decorator is used to add any additional methods or attributes to the functions that the system will find useful.  Note that the decorator does not record this function in any global registry (see below).

Note that if a concrete type has properties, it is necessary to define a "self-translator", which is used translate the value into one with the required properties:

``` python
@metagraph.translator
def nx_to_nx(src: NetworkXGraphType, **props) -> NetworkXGraphType:
    # insert implementation here
```

The `@metagraph.translator` decorator turns the function into a callable object with additional properties:

* `src_type`: ConcreteType class of source
* `dst_type`: ConcreteType class of destination

### Abstract Algorithm

Abstract Algorithms are just Python functions without implementations that have a type signature that includes Abstract Types.  For example, the Louvain community detection might look like this:

``` python
from typing import List

@metagraph.abstract_algorithm(api_group='community')
def louvain(graph: GraphType) -> List[GraphType]):
    '''Return the louvain subgraphs'''
    pass
```

As with the translators, the decorator is used to add useful methods and attributes to the function, as we will see below.

### Concrete Algorithm

Concrete algorithms look like the abstract algorithm, but use concrete types:

``` python
@louvain.concrete_algorithm
def nx_louvain(graph: NetworkXGraphType) -> List[NetworkXGraphType]:
    # insert implementation here
```

Note that this decorator does *not* record the `nx_louvain` method in a registry hidden inside of the abstract `louvain` algorithm.  Instead it converts the function into a callable class with attributes like:

* `nx_louvain.abstract_algorithm`: Reference back to the `louvain` object.
* `nx_louvain.check_args(*args, **kwargs)`: Check if argument list matches function signature.

If we want to define a concrete algorithm that only accepts values with a particular property (allowed properties are enumerated in the concrete type), we can do that this way:

``` python
@louvain.concrete_algorithm
def special_nx_louvain(graph: NetworkGraphXType(foo=True, bar=4)) -> List[NetworkXGraph(foo=True)]:
    # insert implementation here
```

This requires the input `graph` to have both the property of `foo=True` and `bar=4`, and asserts that the return value has property `foo=True`, but nothing else.

### Registration

For both testing purposes, as well as creation of special contexts, we will want to encapsulate the state associated with the registry of types, translators and algorithms.  We call this state a `Resolver`, and it is responsible for:

* Managing a registry of abstract types, concrete types, translators, abstract algorithms, and concrete algorithms.
* Finding a sequence of translators to get between two compatible concrete types
* Selecting a concrete algorithm based on input types and user-specified heuristics.

There will be an implicit, global `Resolver` created by metagraph when imported that is populated by all of the plugins in the environment.  Empty resolvers can also be created and populated manually.

Because translators and concrete algorithms carry references to the abstract types, concrete types, and abstract algorithms they reference, we do not need to explicitly register those with the resolver.  Those can be automatically discovered from the registered translators and concrete algorithms.  (***QUESTION***: is this true??)

The `Resolver` class will have methods like this:

``` python
class Resolver:
    def register(
        self,
        *,
        abstract_types: Optional[List[AbstractType]] = None,
        concrete_types: Optional[List[ConcreteType]] = None,
        translators: Optional[List[Translator]] = None,
        abstract_algorithms: Optional[List[AbstractAlgorithm]] = None,
        concrete_algorithms: Optional[List[ConcreteAlgorithm]] = None,
    ):
        pass

    def load_plugins(self):
        '''Populate registries with plugins from environment'''
        pass

    def typeof(self, obj):
        '''Returns fully specified concrete type of obj'''
        pass

    def convert_to(self, src_obj, dst_type, **props):
        '''Converts src_obj to instance of dst_type with given properties'''
        pass

    def match_algo(self, abstract_algo, arg_types, kwarg_types):
        '''Returns concrete algorithm that matches the given abstract
        algorithm and args/kwargs'''
        pass
```

As a convenience, the resolver can also dynamically generate the algorithm namespace below it.  Ex:

``` python
res = Resolver()
res.load_plugins()

# dispatch and call immediately
mygroups = res.algo.community.louvain(mygraph)

# pick the concrete algo and return it
louvain_func = res.algo.community.louvain.match(mygraph)
```
