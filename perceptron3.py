import numpy as np

entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# saidas = np.array([0, 0, 0, 1])
saidas = np.array([0, 1, 1, 1])  # or
# saidas = np.array([0, 1, 1, 0])  # xor

pesos = np.array([0.0, 0.0])

taxa_de_aprendizagem = 0.1


def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0


def calcula_saida(registros):
    s = registros.dot(pesos)
    return stepFunction(s)


def treinar():
    errototal = 1
    while (errototal != 0):
        errototal = 0
        for i in range(len(saidas)):
            saidaCalculada = calcula_saida(np.array(entradas[i]))
            erro = abs(saidas[i] - saidaCalculada)
            errototal += erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + \
                    (taxa_de_aprendizagem * entradas[i][j] * erro)
                print('peso atualizado ' + str(pesos[j]))
        print('total de erros: ' + str(errototal))


treinar()
print('rede neural treinada')
print(calcula_saida(entradas[0]))
print(calcula_saida(entradas[1]))
print(calcula_saida(entradas[2]))
print(calcula_saida(entradas[3]))
