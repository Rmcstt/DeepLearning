import numpy as np
import matplotlib.pyplot as plt

# Função sigmoid e sua derivada


def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))


def sigmoidDerivada(sig):
    return sig * (1 - sig)


# Dados de entrada e saída (XOR)
entradas = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])

saidas = np.array([[0], [1], [1], [0]])

# Pesos iniciais aleatórios (com pesos para as camadas ocultas)
pesos0 = 2 * np.random.random((2, 3)) - 1
pesos1 = 2 * np.random.random((3, 3)) - 1
pesos2 = 2 * np.random.random((3, 3)) - 1
pesos3 = 2 * np.random.random((3, 3)) - 1
pesos4 = 2 * np.random.random((3, 3)) - 1
pesos5 = 2 * np.random.random((3, 1)) - 1

# Hiperparâmetros
epocas = 999999
taxaAprendizagem = 0.3

# Loop de treinamento
for j in range(epocas):
    # Propagação para frente (Feedforward)
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta0 = sigmoid(somaSinapse0)

    somaSinapse1 = np.dot(camadaOculta0, pesos1)
    camadaOculta1 = sigmoid(somaSinapse1)

    somaSinapse2 = np.dot(camadaOculta1, pesos2)
    camadaOculta2 = sigmoid(somaSinapse2)

    somaSinapse3 = np.dot(camadaOculta2, pesos3)
    camadaOculta3 = sigmoid(somaSinapse3)

    somaSinapse4 = np.dot(camadaOculta3, pesos4)
    camadaOculta4 = sigmoid(somaSinapse4)

    somaSinapse5 = np.dot(camadaOculta4, pesos5)
    camadaSaida = sigmoid(somaSinapse5)

    # Cálculo do erro
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))

    # Propagação para trás (Backpropagation)
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida

    deltaCamadaOculta4 = deltaSaida.dot(
        pesos5.T) * sigmoidDerivada(camadaOculta4)
    deltaCamadaOculta3 = deltaCamadaOculta4.dot(
        pesos4.T) * sigmoidDerivada(camadaOculta3)
    deltaCamadaOculta2 = deltaCamadaOculta3.dot(
        pesos3.T) * sigmoidDerivada(camadaOculta2)
    deltaCamadaOculta1 = deltaCamadaOculta2.dot(
        pesos2.T) * sigmoidDerivada(camadaOculta1)
    deltaCamadaOculta0 = deltaCamadaOculta1.dot(
        pesos1.T) * sigmoidDerivada(camadaOculta0)

    # Atualização dos pesos
    pesos5 += camadaOculta4.T.dot(deltaSaida) * taxaAprendizagem
    pesos4 += camadaOculta3.T.dot(deltaCamadaOculta4) * taxaAprendizagem
    pesos3 += camadaOculta2.T.dot(deltaCamadaOculta3) * taxaAprendizagem
    pesos2 += camadaOculta1.T.dot(deltaCamadaOculta2) * taxaAprendizagem
    pesos1 += camadaOculta0.T.dot(deltaCamadaOculta1) * taxaAprendizagem
    pesos0 += camadaEntrada.T.dot(deltaCamadaOculta0) * taxaAprendizagem

    # Impressão para acompanhar o progresso do treinamento (a cada 1000 épocas)
    if (j % 1000 == 0):
        print(f"Época: {j}, Erro Médio Absoluto: {mediaAbsoluta}")

# Após o treinamento, a rede deve ser capaz de classificar corretamente as entradas XOR
print("\nRede treinada:")
for i in range(len(entradas)):
    camadaEntrada = entradas[i:i+1]  # Pega uma entrada por vez
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta0 = sigmoid(somaSinapse0)

    somaSinapse1 = np.dot(camadaOculta0, pesos1)
    camadaOculta1 = sigmoid(somaSinapse1)

    somaSinapse2 = np.dot(camadaOculta1, pesos2)
    camadaOculta2 = sigmoid(somaSinapse2)

    somaSinapse3 = np.dot(camadaOculta2, pesos3)
    camadaOculta3 = sigmoid(somaSinapse3)

    somaSinapse4 = np.dot(camadaOculta3, pesos4)
    camadaOculta4 = sigmoid(somaSinapse4)

    somaSinapse5 = np.dot(camadaOculta4, pesos5)
    camadaSaida = sigmoid(somaSinapse5)

    print(
        f"Entrada: {entradas[i]}, Saída da Rede: {round(camadaSaida[0][0])}, Saída Esperada: {saidas[i][0]}")

# Visualização dos dados e da linha de decisão
plt.figure(figsize=(10, 6))
plt.scatter(entradas[:, 0], entradas[:, 1], c=saidas, s=100,
            cmap='viridis', edgecolors='k', label='Dados de Entrada (XOR)')
plt.xlabel('Entrada 1')
plt.ylabel('Entrada 2')

# Calcular os pontos da linha de decisão
x1_min, x1_max = entradas[:, 0].min() - 0.5, entradas[:, 0].max() + 0.5
x2_min, x2_max = entradas[:, 1].min() - 0.5, entradas[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                       np.arange(x2_min, x2_max, 0.02))


Z = np.array([sigmoid(
    np.dot(sigmoid(
        np.dot(sigmoid(
            np.dot(sigmoid(
                np.dot(sigmoid(
                    np.dot(sigmoid(
                        np.dot(np.array([x1, x2]), pesos0)),
                        pesos1)),
                       pesos2)),
                   pesos3)),
               pesos4)),
           pesos5))
    for x1, x2 in zip(np.ravel(xx1), np.ravel(xx2))])
Z = Z.reshape(xx1.shape)

# Plotar a linha de decisão
plt.contourf(xx1, xx2, Z, alpha=0.4, cmap='viridis')
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

plt.title('Problema XOR - Perceptron com Múltiplas Camadas Ocultas')
plt.show()
