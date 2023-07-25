from collections.abc import Callable, Mapping
from cls.dsl import Use
from cls.enumeration import enumerate_terms, interpret_term
from cls.fcl import FiniteCombinatoryLogic

from cls.types import Constructor, Literal, Param, Product, TVar, Type


def plus_one(a: str) -> Callable[[Mapping[str, Literal]], int]:
    def _inner(vars: Mapping[str, Literal]) -> int:
        return int(1 + vars[a].value)

    return _inner


def labyrinth() -> None:
    def is_free(a: str, b: str) -> Callable[[Mapping[str, Literal]], bool]:
        return lambda vars: _is_free(
            vars[b].value, vars[a].value
        )  # bool(l_str[vars["b"].value][vars["a"].value] == " ")

    def _is_free(row: int, col: int) -> bool:
        SEED = 0
        if row == col:
            return True
        else:
            return (
                pow(11, (row + col + SEED) * (row + col + SEED) + col + 7, 1000003) % 5
                > 0
            )

    labyrinth_str = [
        " ┃        ",
        " ┃        ",
        " ┃ ┏━━━━ ┓",
        "   ┃     ┃",
        " ┏━┫ ┏━┓ ┗",
        " ┃ ┃ ┃ ┃  ",
        " ┃ ┃ ┗━┻━ ",
        " ┃ ┃      ",
        " ┗━┛ ┏━━┓ ",
        "     ┃  ┃ ",
    ]

    U: Callable[[int, int, int, str], str] = lambda a, _, c, p: f"{p} => UP({c}, {a})"
    D: Callable[[int, int, int, str], str] = lambda _, b, c, p: f"{p} => DOWN({c}, {b})"
    L: Callable[[int, int, int, str], str] = lambda a, _, c, p: f"{p} => LEFT({a}, {c})"
    R: Callable[
        [int, int, int, str], str
    ] = lambda _, b, c, p: f"{p} => RIGHT({b}, {c})"

    pos: Callable[[str, str], Type[str]] = lambda a, b: Constructor(
        "pos", (Product(TVar(a), TVar(b)))
    )

    repo: Mapping[
        Callable[[int, int, int, str], str] | str,
        Param[str] | Type[str],
    ] = {
        U: Use("a", int)
        .Use("b", int)
        .As(plus_one("a"))
        .Use("c", int)
        .With(is_free("c", "a"))
        .Use("pos", pos("c", "b"))
        .In(pos("c", "a")),
        D: Use("a", int)
        .Use("b", int)
        .As(plus_one("a"))
        .Use("c", int)
        .With(is_free("c", "b"))
        .Use("pos", pos("c", "a"))
        .In(pos("c", "b")),
        L: Use("a", int)
        .Use("b", int)
        .As(plus_one("a"))
        .Use("c", int)
        .With(is_free("a", "c"))
        .Use("pos", pos("a", "b"))
        .In(pos("a", "c")),
        R: Use("a", int)
        .Use("b", int)
        .As(plus_one("a"))
        .Use("c", int)
        .With(is_free("b", "c"))
        .Use("pos", pos("a", "c"))
        .In(pos("b", "c")),
        "START": "pos" @ (Literal(0, int) * Literal(0, int)),
    }

    SIZE = 10

    literals = {int: list(range(SIZE))}

    # print("▒▒▒▒▒▒▒▒▒▒▒▒")
    # for line in labyrinth_str:
    #     print(f"▒{line}▒")
    # print("▒▒▒▒▒▒▒▒▒▒▒▒")
    for row in range(SIZE):
        for col in range(SIZE):
            if is_free(row, col):
                print("-", end="")
            else:
                print("#", end="")
        print("")

    fin = "pos" @ (Literal(SIZE - 1, int) * Literal(SIZE - 1, int))

    fcl: FiniteCombinatoryLogic[
        str, Callable[[int, int, int, str], str] | str
    ] = FiniteCombinatoryLogic(repo, literals=literals)

    grammar = fcl.inhabit(fin)

    for term in enumerate_terms(fin, grammar, 3):
        print(interpret_term(term))


if __name__ == "__main__":
    labyrinth()