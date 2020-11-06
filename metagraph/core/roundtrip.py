from .resolver import Resolver, ConcreteType
from .planning import MultiStepTranslator
from .multiverify import MultiVerify, ensure_computed


class UnreachableTranslationError(Exception):
    pass


class RoundTripper:
    def __init__(self, resolver: Resolver):
        self.resolver = resolver
        self.mv = MultiVerify(resolver)

    def _aprops(self, obj):
        obj = ensure_computed(obj)
        return self.resolver.type_of(obj).get_typeinfo(obj).known_abstract_props

    def verify_round_trip(self, obj):
        """
        Verifies that all translators between objects of the same abstract type as `obj` are working correctly.

        The methodology is as follows:
        1. Find all types which are one-hop round-trippable
              Meaning there is a direct translation from `typeof(obj)` to `other_type` and also
              a direct translation from `other_type` back to `typeof(obj)`.
        2. Take `obj` through each one-hop round trip and verify the returned object is equivalent to `obj`
        3. Extend outward in a breadth-first search finding new one-hop translations from the current
              frontier of known good types. The idea is that a two-hop translation has more places which could
              introduce an error, but having checked the one-hop translations first, we can now isolate errors
              to the next ring of translators.
        4. Once all round-trippable steps have been taken for the breadth first search, find any translators
              that were not tested. See if there is a way to exercise the translator and arrive back to the
              starting object (i.e. a circular path). If so, make the translations and check equality.
        5. If any translators are not round-trippable (i.e. not strongly connected with `obj` in the
              translation graph), raise an error. Note that this does not apply to translators across abstract
              types, such as those allowed by unambiguous_subcomponents. Those translators should be exercised
              using `verify_one_way`.
        """
        # Initial sanity check that `obj` is equal to itself
        self.mv.compare_values(
            obj, obj, "roundtrip initial sanity check of obj against itself"
        )
        # Compute properties of `obj`
        ct_obj = self.resolver.typeclass_of(obj)
        at = ct_obj.abstract
        aprops_obj = self._aprops(obj)
        # print(f"Abstract properties of obj:\n{aprops_obj}")

        all_ctypes = {ct for ct in self.resolver.concrete_types if ct.abstract is at}
        # Find all translators with source and target both in abstract type
        translators_to_verify = {}
        for (source, target), func in self.resolver.translators.items():
            if source.abstract is at and target.abstract is at:
                translators_to_verify[(source, target)] = func

        frontier = set()
        known_good_objs = {ct_obj: obj}
        # Expand in breadth-first-search until frontier stops updating
        while known_good_objs.keys() - frontier:
            frontier = set(known_good_objs.keys())
            for ct_target in all_ctypes:
                if ct_target in frontier:
                    continue
                for ct_source in frontier:
                    forward = (ct_source, ct_target)
                    backward = (ct_target, ct_source)
                    if (
                        forward in translators_to_verify
                        and backward in translators_to_verify
                    ):
                        # Found a valid one-hop round trip
                        good_obj = known_good_objs[ct_source]
                        inflight_obj = translators_to_verify[forward](
                            ensure_computed(good_obj), resolver=self.resolver
                        )
                        ct_inflight = self.resolver.typeclass_of(inflight_obj)
                        aprops_inflight = self._aprops(inflight_obj)
                        assert (
                            ct_inflight is ct_target
                        ), f"Translator from {ct_source} to {ct_target} returned an object of type {ct_inflight}"
                        assert (
                            aprops_inflight == aprops_obj
                        ), f"Translated object of type {ct_inflight} has wrong properties {aprops_inflight}"
                        unverified_obj = translators_to_verify[backward](
                            ensure_computed(inflight_obj), resolver=self.resolver
                        )
                        ct_unverified = self.resolver.typeclass_of(unverified_obj)
                        aprops_unverified = self._aprops(unverified_obj)
                        assert (
                            ct_unverified is ct_source
                        ), f"Translator from {ct_target} to {ct_source} returned an object of type {ct_unverified}"
                        assert (
                            aprops_unverified == aprops_obj
                        ), f"Translated object of type {ct_unverified} has wrong properties {aprops_unverified}"
                        # Translate to ct_obj if needed (these are known good translators at this point)
                        if ct_unverified is not ct_obj:
                            unverified_obj = self.resolver.translate(
                                unverified_obj, ct_obj
                            )
                        # Verify equal to `obj`
                        self.mv.compare_values(
                            obj,
                            unverified_obj,
                            f"roundtrip translation from {ct_source} to {ct_target}",
                        )
                        # Remove exercised translators
                        translators_to_verify.pop((ct_source, ct_target))
                        translators_to_verify.pop((ct_target, ct_source))
                        # Add inflight object as a known good object
                        known_good_objs[ct_target] = inflight_obj
        # Look for circular route in unexercised translators
        while translators_to_verify:
            shortest_path = None
            saved_plan = None
            for source, target in translators_to_verify:
                prep_plan = MultiStepTranslator.find_translation(
                    self.resolver, ct_obj, source
                )
                if prep_plan.unsatisfiable:
                    raise UnreachableTranslationError(
                        f"Unable to verify translator from {source} to {target}. Impossible to reach source."
                    )
                return_plan = MultiStepTranslator.find_translation(
                    self.resolver, target, ct_obj
                )
                if return_plan.unsatisfiable:
                    raise UnreachableTranslationError(
                        f"Unable to verify translator from {source} to {target}. Impossible to return from target."
                    )
                # Count translation steps required for full journey
                total_steps = (
                    1 + len(prep_plan.translators) + len(return_plan.translators)
                )
                if saved_plan is None or total_steps < shortest_path:
                    saved_plan = (source, target, prep_plan, return_plan)
                    shortest_path = total_steps
            # Sanity check to avoid a potential infinite loop
            assert shortest_path is not None
            # Only exercise the shortest path
            source, target, prep_plan, return_plan = saved_plan
            prep_obj = prep_plan(obj)
            ct_prep = self.resolver.typeclass_of(prep_obj)
            aprops_prep = self._aprops(prep_obj)
            assert (
                ct_prep is source
            ), f"Translation from {ct_obj} to {source} returned an object of type {ct_prep}"
            assert (
                aprops_prep == aprops_obj
            ), f"Translated object of type {ct_prep} has wrong properties {aprops_prep}"
            inflight_obj = translators_to_verify[(source, target)](
                ensure_computed(prep_obj), resolver=self.resolver
            )
            ct_inflight = self.resolver.typeclass_of(inflight_obj)
            aprops_inflight = self._aprops(inflight_obj)
            assert (
                ct_inflight is target
            ), f"Translation from {source} to {target} returned an object of type {ct_inflight}"
            assert (
                aprops_inflight == aprops_obj
            ), f"Translated object of type {ct_inflight} has wrong properties {aprops_inflight}"
            unverified_obj = return_plan(inflight_obj)
            ct_unverified = self.resolver.typeclass_of(unverified_obj)
            aprops_unverified = self._aprops(unverified_obj)
            assert (
                ct_unverified is ct_obj
            ), f"Translation from {target} to {ct_obj} returned an object of type {ct_unverified}"
            assert (
                aprops_unverified == aprops_obj
            ), f"Translated object of type {ct_unverified} has wrong properties {aprops_unverified}"
            # Verify equal to `obj`
            waypoints = [prep_plan.src_type] + prep_plan.dst_types
            waypoints = waypoints[:-1] + [source, target] + return_plan.dst_types
            self.mv.compare_values(
                obj,
                unverified_obj,
                f"circular trip from {'->'.join(wp.__name__ for wp in waypoints)}",
            )
            # Remove exercised translator
            translators_to_verify.pop((source, target))

    def verify_one_way(self, start, end):
        """
        Verifies that all translators between objects across abstract types (from `start` to `end`)
        are working correctly.

        It is highly recommended to call `verify_round_trip` on both `start` and `end` prior to calling
        `verify_one_way` to have confidence in within-abstract-type translators.

        The methodology is as follows:
        1. Find all translators which go directly from the abstract type of `start` to the abstract type of `end`
        2. For each translator identified,
           a. Translate `start` to the translator's input type
           b. Call the translator
           c. Translate the result to `typeof(end)`
           d. Verify that `end` and the result are equivalent
           e. If the translation from `start` through the translator and on to `end` is not possible (no translators
                 satisfy the required types), raise an error.
        """
        ct_start = self.resolver.typeclass_of(start)
        ct_end = self.resolver.typeclass_of(end)
        at_start = ct_start.abstract
        at_end = ct_end.abstract
        if at_start is at_end:
            raise TypeError(f"start and end must have different abstract types")
        # Find all translators across abstract types
        for (source, target), trans_func in self.resolver.translators.items():
            if source.abstract is at_start and target.abstract is at_end:
                # Find the translation path
                prep_plan = MultiStepTranslator.find_translation(
                    self.resolver, ct_start, source
                )
                if prep_plan.unsatisfiable:
                    raise UnreachableTranslationError(
                        f"Unable to verify translator from {source} to {target}. Impossible to reach source."
                    )
                finish_plan = MultiStepTranslator.find_translation(
                    self.resolver, target, ct_end
                )
                if finish_plan.unsatisfiable:
                    raise UnreachableTranslationError(
                        f"Unable to verify translator from {source} to {target}. Impossible to return from target."
                    )

                # Apply the translations
                prep_obj = prep_plan(start)
                ct_prep = self.resolver.typeclass_of(prep_obj)
                assert (
                    ct_prep is source
                ), f"Translation from {ct_start} to {source} returned an object of type {ct_prep}"
                inflight_obj = trans_func(
                    ensure_computed(prep_obj), resolver=self.resolver
                )
                ct_inflight = self.resolver.typeclass_of(inflight_obj)
                assert (
                    ct_inflight is target
                ), f"Translation from {source} to {target} returned an object of type {ct_inflight}"
                unverified_obj = finish_plan(inflight_obj)
                ct_unverified = self.resolver.typeclass_of(unverified_obj)
                assert (
                    ct_unverified is ct_end
                ), f"Translation from {target} to {ct_end} returned an object of type {ct_unverified}"
                # Check for equivalency with `end`
                waypoints = [prep_plan.src_type] + prep_plan.dst_types
                waypoints = waypoints[:-1] + [source, target] + finish_plan.dst_types
                self.mv.compare_values(
                    end,
                    unverified_obj,
                    f"one-way trip from {'->'.join(wp.__name__ for wp in waypoints)}",
                )
