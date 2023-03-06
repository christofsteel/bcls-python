from collections import deque
from collections.abc import Hashable, Iterable
from itertools import chain
from typing import Generic, TypeVar

from .types import Arrow, Constructor, Intersection, Product, Type

T = TypeVar("T", bound=Hashable, covariant=True)

def flatten_gen(subtypes):
    for subtype in subtypes:
        if isinstance(subtype, Intersection):
            for elem in subtype.inner:
                yield elem
        else:
            yield subtype


class Subtypes(Generic[T]):
    def __init__(self, environment: dict[T, set[T]]):
        self.environment = self._transitive_closure(
            self._reflexive_closure(environment)
        )

    def _check_subtype_rec(self, subtypes: Iterable[Type[T]], supertype: Type[T]) -> bool:
        if supertype.is_omega:
            return True
        # st = chain(subtype.inner if isinstance(subtype, Intersection) else [subtype] for subtype in subtypes)
        st = flatten_gen(subtypes)
        # st = deque()
        # for subtype in subtypes:
        #     if isinstance(subtype, Intersection):
        #         st.extend(subtype.inner)
        #     else:
        #         st.append(subtype)
        match supertype:
            case Constructor(name2, arg2):
                casted_constr: Iterable[Type[T]] = list(map(lambda s:s.arg, filter(lambda subtype: isinstance(subtype, Constructor) and (subtype.name == name2 or (subtype.name in self.environment and name2 in self.environment[subtype.name])), st)))
                # casted_constr: deque[Type[T]] = deque()
                # for subtype in st:
                #     if isinstance(subtype, Constructor) and (subtype.name == name2 or (subtype.name in self.environment and name2 in self.environment[subtype.name])):
                #         casted_constr.append(subtype.arg)

                return len(casted_constr) != 0 and self._check_subtype_rec(
                    casted_constr, arg2
                )
            case Arrow(src2, tgt2):
                casted_arr: deque[Type[T]] = deque()
                for subtype in st:
                    if isinstance(subtype, Arrow) and self._check_subtype_rec(deque(src2), subtype.source):
                        casted_arr.append(subtype.target)
                return len(casted_arr) != 0 and self._check_subtype_rec(
                    casted_arr, tgt2
                )
            case Product(l2, r2):
                casted_l: deque[Type[T]] = deque()
                casted_r: deque[Type[T]] = deque()
                for subtype in st:
                    if isinstance(subtype, Product):
                        casted_l.append(subtype.left)
                        casted_r.append(subtype.right)
                return (
                    len(casted_l) != 0
                    and len(casted_r) != 0
                    and self._check_subtype_rec(casted_l, l2)
                    and self._check_subtype_rec(casted_r, r2)
                )
            case Intersection(inner):
                return all(self._check_subtype_rec(subtypes, subformula) for subformula in inner)
            case _:
                raise TypeError(f"Unsupported type in check_subtype: {supertype}")

    def check_subtype(self, subtype: Type[T], supertype: Type[T]) -> bool:
        """Decides whether subtype <= supertype."""

        return self._check_subtype_rec(deque((subtype,)), supertype)

    @staticmethod
    def _reflexive_closure(env: dict[T, set[T]]) -> dict[T, set[T]]:
        all_types: set[T] = set(env.keys())
        for v in env.values():
            all_types.update(v)
        result: dict[T, set[T]] = {
            subtype: {subtype}.union(env.get(subtype, set())) for subtype in all_types
        }
        return result

    @staticmethod
    def _transitive_closure(env: dict[T, set[T]]) -> dict[T, set[T]]:
        result: dict[T, set[T]] = {
            subtype: supertypes.copy() for (subtype, supertypes) in env.items()
        }
        has_changed = True

        while has_changed:
            has_changed = False
            for known_supertypes in result.values():
                for supertype in known_supertypes.copy():
                    to_add: set[T] = {
                        new_supertype
                        for new_supertype in result[supertype]
                        if new_supertype not in known_supertypes
                    }
                    if to_add:
                        has_changed = True
                    known_supertypes.update(to_add)

        return result

    def minimize(self, tys: set[Type[T]]) -> set[Type[T]]:
        result: set[Type[T]] = set()
        for ty in tys:
            if all(map(lambda ot: not self.check_subtype(ot, ty), result)):
                result = {ty, *(ot for ot in result if not self.check_subtype(ty, ot))}
        return result
