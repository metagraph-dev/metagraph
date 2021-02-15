import dask.base
from .tasks import MetagraphTask, DelayedAlgo, DelayedJITAlgo, DelayedTranslate
from copy import deepcopy
from typing import Dict, Hashable, Any
from collections.abc import Mapping
from metagraph.core.compiler import optimize


def merge_dict_of_dict(
    base: Dict[Hashable, Dict[Hashable, Any]],
    overlay: Dict[Hashable, Dict[Hashable, Any]],
):
    """Returns a copy of base with keys and values from overlay merged on top."""
    result = deepcopy(base)
    for key, attrs in overlay.items():
        if key not in result:
            result[key] = attrs.copy()
        else:
            result[key].update(attrs)
    return result


def visualize(*dags, filename="mydask", format=None, optimize_graph=False, **kwargs):
    """Custom visualization of DAGs with Metagraph nodes.

    Arguments are the same as standard dask visualize method"""

    # We customize the behavior of the visualization entirely through function
    # and data attributes. Function attributes style the node corresponding to
    # the task execution node whereas data attributes style the output result
    # node associated with the task.  Result nodes can be hidded with the
    # ``collapse_ouput`` option. Attribute key/value pairs can be anything
    # that graphviz can handle.

    # combine arguments into one large task list (cf. dask.base.visualize)
    merged_dag = {}
    output_keys = set()
    for dag in dags:
        if isinstance(dag, Mapping):
            merged_dag.update(dag)
        elif dask.base.is_dask_collection(dag):
            merged_dag.update(dag.__dask_graph__())
            output_keys = output_keys.union(set(dag.__dask_keys__()))

    if optimize_graph:
        merged_dag = optimize(merged_dag, output_keys=list(output_keys))

    # To give the caller priority to override styling, first compute
    # attributes then overlay any attributes that were passed in.
    function_attributes = {}
    data_attributes = {}
    for key, task in merged_dag.items():
        task_callable = task[0]

        if isinstance(task_callable, MetagraphTask):
            func_attrs = {
                "label": task_callable.func_label,
            }
            data_attrs = {
                "shape": "parallelogram",
                "label": task_callable.data_label,
            }

            if isinstance(task_callable, DelayedAlgo):
                func_attrs["shape"] = "octagon"
                if task_callable.algo._compiler is not None:
                    func_attrs["color"] = "red"
                    func_attrs["penwidth"] = "2.0"
            if isinstance(task_callable, DelayedTranslate):
                func_attrs["shape"] = "ellipse"
            if isinstance(task_callable, DelayedJITAlgo):
                func_attrs["shape"] = "doubleoctagon"
                func_attrs["color"] = "red"
                func_attrs["penwidth"] = "2.0"
        else:
            continue

        function_attributes[key] = func_attrs
        data_attributes[key] = data_attrs

    # overlay user-provided attributes
    user_function_attributes = kwargs.pop("function_attributes", {})
    user_data_attributes = kwargs.pop("data_attributes", {})

    # let dask's visualize() do the heavy lifting
    return dask.base.visualize(
        merged_dag,
        filename=filename,
        format=format,
        optimize_graph=optimize_graph,
        function_attributes=merge_dict_of_dict(
            function_attributes, user_function_attributes
        ),
        data_attributes=merge_dict_of_dict(data_attributes, user_data_attributes),
        **kwargs,
    )
