# Propositional Finite Combinatory Logic

from __future__ import annotations
from collections import deque
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, Generic, Optional, TypeAlias, TypeVar, reveal_type

from .combinatorics import maximal_elements, minimal_covers, partition
from .subtypes import Subtypes
from .types import Arrow, Intersection, Literal, Pi, Predicate, TVar, Type

T = TypeVar("T", bound=Hashable, covariant=True)
C = TypeVar("C")

Ter = TypeVar("Ter")
NT = TypeVar("NT")
Ann = TypeVar("Ann")

# ([sigma_1, ..., sigma_n], tau) means sigma_1 -> ... -> sigma_n -> tau
MultiArrow: TypeAlias = tuple[list[Type[T]], Type[T]]

dict_keys = type({}.keys())

TreeGrammar: TypeAlias = MutableMapping[Type[T], deque[tuple[C, list[Type[T]]]]]


@dataclass()
class AnnotatedRHS(Generic[Ter, NT, Ann]):
    terminal: Ter
    args: list[NT]
    annotation: Ann

    def __str__(self):
        return f"{self.annotation}=>{self.terminal} {' '.join([str(arg) for arg in self.args])}"


class AnnotatedTreeGrammar(Generic[Ter, NT, Ann]):
    def __init__(self):
        self.rules: dict[NT, deque[AnnotatedRHS[Ter, NT, Ann]]] = {}

    def add_rule(self, non_terminal: NT, rhs: AnnotatedRHS[Ter, NT, Ann]):
        if non_terminal in self.rules:
            self.rules[non_terminal].append(rhs)
        else:
            self.rules[non_terminal] = deque([rhs])

    def __getitem__(self, non_terminal: NT) -> deque[AnnotatedRHS[Ter, NT, Ann]]:
        return self.rules[non_terminal]

    def __setitem__(self, non_terminal: NT, rhs: deque[AnnotatedRHS[Ter, NT, Ann]]):
        self.rules[non_terminal] = rhs

    def get(self, non_terminal: NT) -> Optional[deque[AnnotatedRHS[Ter, NT, Ann]]]:
        if non_terminal in self.rules:
            return self[non_terminal]
        return None

    def non_terminals(self) -> Iterable[NT]:
        return self.rules.keys()

    def all_rules(self) -> Iterable[tuple[NT, deque[AnnotatedRHS[Ter, NT, Ann]]]]:
        return self.rules.items()

    def show(self) -> str:
        output = ""
        for lhs, rhss in self.rules.items():
            output += str(lhs) + " |-> " + " | ".join([str(rhs) for rhs in rhss]) + "\n"
        return output


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
        instantiated_repository = {
            combinator: (
                typ
                if isinstance(typ, Type)
                else reduce(Intersection, typ.multi_instantiate_all_rec(literals))
            )
            for combinator, typ in repository.items()
        }

        self.literals = literals

        self.repository: Mapping[C, list[list[MultiArrow[T]]]] = {
            c: list(FiniteCombinatoryLogic._function_types(ty))
            for c, ty in instantiated_repository.items()
        }
        self.subtypes = subtypes

    # @staticmethod
    # def _instantiate(repository: Mapping[C, Pi[T] | Type[T]]) -> Mapping[C, Pi[T]]:
    #     instantiated_repository = {}
    #     for combinator, pi_or_type in repository.items():
    #         if isinstance(pi_or_type, Pi):
    #             typed_parameters_subst = pi_or_type.multi_instantiate_all_rec(literals)
    #             # {
    #             #     name: TVar(name, typ) for name, typ in pi_or_type.parameters
    #             # }
    #             instantiated_repository[combinator] = pi_or_type.pi_subst(
    #                 typed_parameters_subst
    #             )
    #         else:
    #             instantiated_repository[combinator] = Pi([], pi_or_type)
    #     return instantiated_repository

    @staticmethod
    def _function_types(ty: Pi[T]) -> Iterable[list[MultiArrow[T]]]:
        """Presents a type as a list of 0-ary, 1-ary, ..., n-ary function types."""

        def unary_function_types(ty: Type[T]) -> Iterable[tuple[Type[T], Type[T]]]:
            tys: deque[Type[T]] = deque((ty,))
            while tys:
                match tys.pop():
                    case Arrow(src, tgt) if not tgt.is_omega:
                        yield (src, tgt)
                    case Intersection(sigma, tau):
                        tys.extend((sigma, tau))

        current: list[MultiArrow[T]] = [([], ty.type)]
        while len(current) != 0:
            yield current
            current = [
                (args + [new_arg], new_tgt)
                for (args, tgt) in current
                for (new_arg, new_tgt) in unary_function_types(tgt)
            ]

    def _subqueries(
        self,
        nary_types: list[MultiArrow[T]],
        paths: list[Type[T]],
        # parameters: list[tuple[str, Any]],
    ) -> Sequence[list[Type[T]]]:
        # does the target of a multi-arrow contain a given type?
        target_contains: Callable[
            [MultiArrow[T], Type[T]], bool
        ] = lambda m, t: self.subtypes.check_subtype(m[1], t)
        # cover target using targets of multi-arrows in nary_types
        covers = minimal_covers(nary_types, paths, target_contains)
        # reveal_type(covers)
        if len(covers) == 0:
            return []
        # intersect corresponding arguments of multi-arrows in each cover
        intersect_args: Callable[
            [Iterable[Type[T]], Iterable[Type[T]]], list[Type[T]]
        ] = lambda args1, args2: [Intersection(a, b) for a, b in zip(args1, args2)]

        intersected_args = [
            list(reduce(intersect_args, (m[0] for m in ms))) for ms in covers
        ]
        intersected_target = [[m[1] for m in ms] for ms in covers]
        # reveal_type(intersected_target)
        # consider only maximal argument vectors
        compare_args = lambda args1, args2: all(
            map(self.subtypes.check_subtype, args1, args2)
        )

        print(list(intersected_args))
        print(intersected_target)
        return maximal_elements(intersected_args, compare_args)

    def inhabit(self, *targets: Type[T]) -> AnnotatedTreeGrammar[C, Type[T], Predicate]:
        type_targets = deque(targets)

        # dictionary of type |-> sequence of combinatory expressions
        memo: AnnotatedTreeGrammar[C, Type[T], Predicate] = AnnotatedTreeGrammar()

        while type_targets:
            current_target = type_targets.pop()
            if memo.get(current_target) is None:
                # target type was not seen before
                # paths: list[Type] = list(target.organized)
                # possibilities: deque[tuple[C, Predicate, Sequence[tuple[Any, Any]], list[Type[T]]]] = deque()
                # memo.add_rule(current_target, possibilities)
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
                            print(
                                f"Combinator: {combinator}\nSQ: {subquery}\n\tArguments: {arguments}\n\tNary: {nary_types}\n\t\tPaths: {paths}"
                            )
                            memo.add_rule(
                                current_target,
                                AnnotatedRHS(
                                    combinator, subquery, self.metadata[combinator][1]
                                ),
                            )
                            # possibilities.append((combinator, self.metadata[combinator][1], self.metadata[combinator][0],subquery))
                            type_targets.extendleft(subquery)

        # prune not inhabited types
        FiniteCombinatoryLogic._prune(memo)

        return memo

    @staticmethod
    def _prune(memo: AnnotatedTreeGrammar[C, Type[T], Predicate]) -> None:
        """Keep only productive grammar rules."""

        def is_ground(args: list[Type[T]], ground_types: set[Type[T]]) -> bool:
            return all(True for arg in args if arg in ground_types)

        ground_types: set[Type[T]] = set()
        new_ground_types, candidates = partition(
            lambda ty: any(
                True for rhs in memo[ty] if is_ground(rhs.args, ground_types)
            ),
            memo.non_terminals(),
        )
        # initialize inhabited (ground) types
        while new_ground_types:
            ground_types.update(new_ground_types)
            new_ground_types, candidates = partition(
                lambda ty: any(
                    True for rhs in memo[ty] if is_ground(rhs.args, ground_types)
                ),
                candidates,
            )

        for target, possibilities in memo.rules.items():
            memo[target] = deque(
                possibility
                for possibility in possibilities
                if is_ground(possibility.args, ground_types)
            )
