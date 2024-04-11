
import numpy as np

entrada = np.array([1, 7, 5])

pesos = np.array([0.8, 0.1, 0])


def soma(e, p):

    return e.dot(p)


s = soma(entrada, pesos)


def stepFunction(s):
    if s >= 1:
        return 1
    return 0


print(stepFunction(s))
