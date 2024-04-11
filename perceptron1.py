

entrada = [-1, 7, 5]

pesos = [0.8, 0.1, 0]


def soma(e, p):
    s = 0
    for i in range(3):
        # print(entrada[i])
        # print(pesos[i])
        s += e[i] * p[i]
    return s


s = soma(entrada, pesos)


def stepFunction(s):
    if s >= 1:
        return 1
    return 0


print(stepFunction(s))
