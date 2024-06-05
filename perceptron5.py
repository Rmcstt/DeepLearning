import numpy as np


# sigmoid


def sigmoid(soma):
    return 1 / (1 + np.exp(- soma))


def sigmoidDerivada(sig):
    return sig * (1 - sig)


# a = sigmoid(-1.43)
# b = sigmoidDerivada(a)

# print(a)
# print(b)

# a = np.exp(-7)
# b = sigmoid(-1)

entradas = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])

saidas = np.array([[0], [1], [1], [0]])

# pesos0 = np.array([[-0.424, -0.740, -0.961],
#                    [0.358, -0.577, -0.469]])

# pesos1 = np.array([[-0.017], [-0.893], [0.148]])

pesos0 = 2*np.random.random((2, 3)) - 1
pesos1 = 2*np.random.random((3, 1)) - 1

epocas = 10000
taxaAprendizagem = 0.5
momento = 1

for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)

    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)

    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print(f'epoca: {j} ' + 'erro ' + str(mediaAbsoluta))

    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida

    pesos1Transposta = pesos1.T
    deltaSaidaPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaPeso * sigmoidDerivada(camadaOculta)

    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)

    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)


# print(pesos1)
# [[-0.00711903]
# [-0.88642447]
# [ 0.15432644]]

# print(camadaOculta)
#  [0.5        0.5        0.5       ]
#  [0.5885562  0.35962319 0.38485296]
#  [0.39555998 0.32300414 0.27667802]
#  [0.48350599 0.21131785 0.19309868]


# print(somaSinapse0)
#  [ 0.     0.     0.   ]
#  [ 0.358 -0.577 -0.469]
#  [-0.424 -0.74  -0.961]
#  [-0.066 -1.317 -1.43 ]


# print(somaSinapse1)
# [-0.381     ]
# [-0.27419072]
# [-0.25421887]
# [-0.16834784]

# print(camadaSaida)
# [0.40588573]
# [0.43187857]
# [0.43678536]
# [0.45801216]

# print(erroCamadaSaida)
# [-0.40588573]
# [ 0.56812143]
# [ 0.56321464]
# [-0.45801216]

# print(mediaAbsoluta)
# 0.49880848923713045


# print(derivadaSaida)
# [0.2411425 ]
# [0.24535947]
# [0.24600391]
# [0.24823702]

# print(deltaSaida)
# [-0.0978763 ]
# [ 0.13939397]
# [ 0.138553  ]
# [-0.11369557]
