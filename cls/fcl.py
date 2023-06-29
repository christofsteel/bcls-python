# Propositional Finite Combinatory Logic

from collections import deque
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from functools import reduce
from typing import Callable, Generic, TypeAlias, TypeVar

from .combinatorics import maximal_elements, minimal_covers, partition
from .subtypes import Subtypes
from .types import Arrow, Intersection, Literal, Pi, Type

T = TypeVar("T", bound=Hashable, covariant=True)
C = TypeVar("C")

# ([sigma_1, ..., sigma_n], tau) means sigma_1 -> ... -> sigma_n -> tau
MultiArrow: TypeAlias = tuple[list[Type[T]], Type[T]]


TreeGrammar: TypeAlias = MutableMapping[Type[T], deque[tuple[C, list[Type[T]]]]]


def show_grammar(grammar: TreeGrammar[T, C]) -> Iterable[str]:
    for clause, possibilities in grammar.items():
        lhs = str(clause)
        yield (
            lhs
            + " => "
            + "; ".join(
                (str(combinator) + "(" + ", ".join(map(str, args)) + ")")
                for combinator, args in possibilities
            )
        )


def mstr(m: MultiArrow[T]) -> tuple[str, str]:
    return (str(list(map(str, m[0]))), str(m[1]))


class FiniteCombinatoryLogic(Generic[T, C]):
    def __init__(
        self,
        repository: Mapping[C, Pi[T] | Type[T]],
        literals: Sequence[Literal[T]],
        subtypes: Subtypes[T],
    ):
        instantiated_repository = FiniteCombinatoryLogic._instantiate(
            repository, literals
        )

        self.repository: Mapping[C, list[list[MultiArrow[T]]]] = {
            c: list(FiniteCombinatoryLogic._function_types(ty))
            for c, ty in instantiated_repository.items()
        }
        self.subtypes = subtypes

    @staticmethod
    def _instantiate(
        repository: Mapping[C, Pi[T] | Type[T]], literals: Sequence[Literal[T]]
    ) -> Mapping[C, Type[T]]:
        instantiated_repository = {}
        for combinator, pi_or_type in repository.items():
            if isinstance(pi_or_type, Pi):
                instantiated_repository[combinator] = reduce(
                    Intersection, pi_or_type.instantiate_all(literals)
                )
            else:
                instantiated_repository[combinator] = pi_or_type
        for literal in literals:
            instantiated_repository[literal.value] = literal
        return instantiated_repository

    @staticmethod
    def _function_types(ty: Type[T]) -> Iterable[list[MultiArrow[T]]]:
        """Presents a type as a list of 0-ary, 1-ary, ..., n-ary function types."""

        def unary_function_types(ty: Type[T]) -> Iterable[tuple[Type[T], Type[T]]]:
            tys: deque[Type[T]] = deque((ty,))
            while tys:
                match tys.pop():
                    case Arrow(src, tgt) if not tgt.is_omega:
                        yield (src, tgt)
                    case Intersection(sigma, tau):
                        tys.extend((sigma, tau))

        current: list[MultiArrow[T]] = [([], ty)]
        while len(current) != 0:
            yield current
            current = [
                (args + [new_arg], new_tgt)
                for (args, tgt) in current
                for (new_arg, new_tgt) in unary_function_types(tgt)
            ]

    def _subqueries(
        self, nary_types: list[MultiArrow[T]], paths: list[Type[T]]
    ) -> Sequence[list[Type[T]]]:
        # does the target of a multi-arrow contain a given type?
        target_contains: Callable[
            [MultiArrow[T], Type[T]], bool
        ] = lambda m, t: self.subtypes.check_subtype(m[1], t)
        # cover target using targets of multi-arrows in nary_types
        covers = minimal_covers(nary_types, paths, target_contains)
        if len(covers) == 0:
            return []
        # intersect corresponding arguments of multi-arrows in each cover
        intersect_args: Callable[
            [Iterable[Type[T]], Iterable[Type[T]]], list[Type[T]]
        ] = lambda args1, args2: [Intersection(a, b) for a, b in zip(args1, args2)]

        intersected_args = (
            list(reduce(intersect_args, (m[0] for m in ms))) for ms in covers
        )
        # consider only maximal argument vectors
        compare_args = lambda args1, args2: all(
            map(self.subtypes.check_subtype, args1, args2)
        )
        return maximal_elements(intersected_args, compare_args)

    def inhabit(self, *targets: Type[T]) -> TreeGrammar[T, C]:
        type_targets = deque(targets)

        # dictionary of type |-> sequence of combinatory expressions
        memo: TreeGrammar[T, C] = dict()

        while type_targets:
            current_target = type_targets.pop()
            if memo.get(current_target) is None:
                # target type was not seen before
                # paths: list[Type] = list(target.organized)
                possibilities: deque[tuple[C, list[Type[T]]]] = deque()
                memo.update({current_target: possibilities})
                # If the target is omega, then the result is junk
                if current_target.is_omega:
                    continue

                paths: list[Type[T]] = list(current_target.organized)

                # try each combinator and arity
                for combinator, combinator_type in self.repository.items():
                    for nary_types in combinator_type:
                        arguments: list[list[Type[T]]] = list(
                            self._subqueries(nary_types, paths)
                        )
                        if len(arguments) == 0:
                            continue

                        for subquery in arguments:
                            possibilities.append((combinator, subquery))
                            type_targets.extendleft(subquery)

        # prune not inhabited types
        FiniteCombinatoryLogic._prune(memo)

        return memo

    @staticmethod
    def _prune(memo: TreeGrammar[T, C]) -> None:
        """Keep only productive grammar rules."""

        def is_ground(args: list[Type[T]], ground_types: set[Type[T]]) -> bool:
            return all(True for arg in args if arg in ground_types)

        ground_types: set[Type[T]] = set()
        new_ground_types, candidates = partition(
            lambda ty: any(
                True for (_, args) in memo[ty] if is_ground(args, ground_types)
            ),
            memo.keys(),
        )
        # initialize inhabited (ground) types
        while new_ground_types:
            ground_types.update(new_ground_types)
            new_ground_types, candidates = partition(
                lambda ty: any(
                    True for _, args in memo[ty] if is_ground(args, ground_types)
                ),
                candidates,
            )

        for target, possibilities in memo.items():
            memo[target] = deque(
                possibility
                for possibility in possibilities
                if is_ground(possibility[1], ground_types)
            )
