from cls import inhabit_and_interpret
from cls.types import Literal, Pi, Constructor, Arrow, TVar


def X(y):
    return f"X({y})"


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

    l = inhabit_and_interpret(Gamma, query, literals=Delta)
    for x in l:
        print(f"Query: {query}")
        print(x)


if __name__ == "__main__":
    main()
