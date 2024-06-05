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

# Pesos iniciais aleatórios (com pesos para a camada oculta)

pesos0 = 2 * np.random.random((2, 3)) - 1
pesos1 = 2 * np.random.random((3, 1)) - 1

# Hiperparâmetros
epocas = 100000
taxaAprendizagem = 0.9

# Loop de treinamento
for j in range(epocas):
    # Propagação para frente (Feedforward)
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)

    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)

    # Cálculo do erro
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))

    # Propagação para trás (Backpropagation)
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida

    deltaCamadaOculta = deltaSaida.dot(
        pesos1.T) * sigmoidDerivada(camadaOculta)

    # Atualização dos pesos
    pesos1 += camadaOculta.T.dot(deltaSaida) * taxaAprendizagem
    pesos0 += camadaEntrada.T.dot(deltaCamadaOculta) * taxaAprendizagem

    # Impressão para acompanhar o progresso do treinamento (a cada 1000 épocas)
    if (j % 1000 == 0):
        print(f"Época: {j}, Erro Médio Absoluto: {mediaAbsoluta}")

# Após o treinamento, a rede deve ser capaz de classificar corretamente as entradas XOR
print("\nRede treinada:")
for i in range(len(entradas)):
    camadaEntrada = entradas[i:i+1]  # Pega uma entrada por vez
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    print(
        f"Entrada: {entradas[i]}, Saída da Rede: {round(camadaSaida[0][0])}, Saída Esperada: {saidas[i][0]}")

# Visualização dos dados e da linha de decisão
plt.scatter(entradas[:, 0], entradas[:, 1], c=saidas, s=100, cmap='viridis')
plt.xlabel('Entrada 1')
plt.ylabel('Entrada 2')

# Calcular os pontos da linha de decisão
x1_min, x1_max = entradas[:, 0].min() - 0.5, entradas[:, 0].max() + 0.5
x2_min, x2_max = entradas[:, 1].min() - 0.5, entradas[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                       np.arange(x2_min, x2_max, 0.02))

Z = np.array([sigmoid(np.dot(sigmoid(np.dot(np.array([x1, x2]), pesos0)), pesos1))
             for x1, x2 in zip(np.ravel(xx1), np.ravel(xx2))])
Z = Z.reshape(xx1.shape)

# Plotar a linha de decisão
plt.contourf(xx1, xx2, Z, alpha=0.4, cmap='viridis')
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

plt.title('Problema XOR - Perceptron com uma Camada Oculta')
plt.show()
