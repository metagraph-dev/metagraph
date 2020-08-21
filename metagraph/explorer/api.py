import json
import asyncio

# --------------------------
# PoC Code from https://websockets.readthedocs.io/en/stable/intro.html
# --------------------------
class Valuable:
    def __init__(self):
        self.STATE = {"value": 0}

    def adjust_value(self, amount):
        self.STATE["value"] += amount

    def state_event(self):
        return json.dumps({"type": "state", **self.STATE})

    async def notify_state(self, connections):
        if connections:  # asyncio.wait doesn't accept an empty list
            message = self.state_event()
            await asyncio.wait([user.send(message) for user in connections])


# ----------------------------

# Guiding principles
# 1. Make API functions testable in Python
# 2. API functions should return Python objects which are easily converted into JSON (dict, list, str, int, bool)
# 3. Make the server handle all conversion to/from JSON
# 4. Make object structure as consistent as possible so decoding on Javascript side is easier


def list_plugins(resolver):
    # This is tricky because we can find the plugins for the default resolver, but not for a custom resolver
    raise NotImplementedError()


def list_types(resolver, filters):
    raise NotImplementedError()


def list_translators(resolver, source_type, filters):
    raise NotImplementedError()


def list_algorithms(resolver, filters):
    raise NotImplementedError()


def list_algorithm_params(resolver, abstract_pathname):
    raise NotImplementedError()


def solve_translator(resolver, src_type, dst_type):
    raise NotImplementedError()


def solve_algorithm(resolver, abstract_pathname, params, returns):
    raise NotImplementedError()
