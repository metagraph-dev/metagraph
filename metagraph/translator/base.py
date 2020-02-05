from functools import wraps

"""
All potential properties of the input class and
all potential properties of the output class must
be handled by the translator.
Partial implementations are not allowed.

Only one translator can be registered for each
input/output class combination.

Try translator function signature:
    func(x, **props)
Given object x (will be of type indicated in `input_class`),
must return an object of type indicated in `output_class`
with properties matching those indicated in `props`.
"""

_translator_registry = {}


def register_translator(input_class, output_class):
    def inner_decorator(func):
        @wraps(func)
        def translator_wrapper(x, **props):
            return func(x, **props)

        # Register this function with the registry
        key = (input_class, output_class)
        if key in _translator_registry:
            raise Exception(
                f"Attempt to register more than one function for {input_class}->{output_class} translation"
            )
        _translator_registry[key] = translator_wrapper
        return translator_wrapper

    return inner_decorator
