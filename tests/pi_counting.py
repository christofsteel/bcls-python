from collections import deque
from cls import inhabit_and_interpret, FiniteCombinatoryLogic
from cls.enumeration import enumerate_terms, interpret_term
from cls.fcl import AnnotatedRHS
from cls.subtypes import Subtypes
from cls.types import Literal, Pi, Constructor, Arrow, TVar


def X(y):
    return f"X<a=,b=>({y})"


def Y():
    return "Y"


def main():
    P = lambda vars: vars["a"] + 1 == vars["b"]
    Gamma = {
        X: Pi(
            [("a", int), ("b", int)],
            Arrow(Constructor("c", TVar("a")), Constructor("c", TVar("b"))),
            P,
        ),
        Y: Constructor("c", Literal(3, int)),
    }
    Delta = [Literal(i, int) for i in range(11)]

    query = Constructor("c", Literal(5, int))

    # l = inhabit_and_interpret(Gamma, query, literals=Delta)
    l = FiniteCombinatoryLogic(Gamma, Delta, Subtypes({})).inhabit(query)
    print(l.show())
    for x in enumerate_terms(query, l):
        print(f"Query: {query}")
        print(interpret_term(x))
        break


if __name__ == "__main__":
    main()
