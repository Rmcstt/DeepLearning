import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Função ReLU e sua derivada


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Função sigmoid e sua derivada


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(sig):
    return sig * (1 - sig)


# Carregar o dataset de câncer de mama
base = datasets.load_breast_cancer()
entradas = base.data
valores_saida = base.target
saidas = valores_saida.reshape(-1, 1)  # Redimensionar para vetor coluna

# Normalizar as entradas
scaler = StandardScaler()
entradas = scaler.fit_transform(entradas)

# Inicializar pesos com He initialization


def initialize_weights_he(l_in, l_out):
    return np.random.randn(l_in, l_out) * np.sqrt(2. / l_in)


pesos0 = initialize_weights_he(30, 10)
pesos1 = initialize_weights_he(10, 10)
pesos2 = initialize_weights_he(10, 10)
pesos3 = initialize_weights_he(10, 1)

# Hiperparâmetros
epocas = 100000
taxa_aprendizagem = 0.01

# Loop de treinamento
for j in range(epocas):
    # Propagação para frente (Feedforward)
    camada_entrada = entradas
    soma_sinapse0 = np.dot(camada_entrada, pesos0)
    camada_oculta1 = relu(soma_sinapse0)

    soma_sinapse1 = np.dot(camada_oculta1, pesos1)
    camada_oculta2 = relu(soma_sinapse1)

    soma_sinapse2 = np.dot(camada_oculta2, pesos2)
    camada_oculta3 = relu(soma_sinapse2)

    soma_sinapse3 = np.dot(camada_oculta3, pesos3)
    camada_saida = sigmoid(soma_sinapse3)

    # Calcular erro
    erro_camada_saida = saidas - camada_saida
    media_absoluta = np.mean(np.abs(erro_camada_saida))

    # Retropropagação (Backpropagation)
    derivada_saida = sigmoid_derivative(camada_saida)
    delta_saida = erro_camada_saida * derivada_saida

    delta_camada_oculta3 = delta_saida.dot(
        pesos3.T) * relu_derivative(camada_oculta3)
    delta_camada_oculta2 = delta_camada_oculta3.dot(
        pesos2.T) * relu_derivative(camada_oculta2)
    delta_camada_oculta1 = delta_camada_oculta2.dot(
        pesos1.T) * relu_derivative(camada_oculta1)

    # Atualizar pesos
    pesos3 += camada_oculta3.T.dot(delta_saida) * taxa_aprendizagem
    pesos2 += camada_oculta2.T.dot(delta_camada_oculta3) * taxa_aprendizagem
    pesos1 += camada_oculta1.T.dot(delta_camada_oculta2) * taxa_aprendizagem
    pesos0 += camada_entrada.T.dot(delta_camada_oculta1) * taxa_aprendizagem

    # Imprimir erro a cada 1000 épocas
    if j % 1000 == 0:
        print(f"Época: {j}, Erro Médio Absoluto: {media_absoluta}")

# Após o treinamento, a rede deve classificar corretamente as entradas
print("\nRede treinada:")
for i in range(len(entradas)):
    camada_entrada = entradas[i:i+1]  # Pegar uma entrada por vez
    soma_sinapse0 = np.dot(camada_entrada, pesos0)
    camada_oculta1 = relu(soma_sinapse0)

    soma_sinapse1 = np.dot(camada_oculta1, pesos1)
    camada_oculta2 = relu(soma_sinapse1)

    soma_sinapse2 = np.dot(camada_oculta2, pesos2)
    camada_oculta3 = relu(soma_sinapse2)

    soma_sinapse3 = np.dot(camada_oculta3, pesos3)
    camada_saida = sigmoid(soma_sinapse3)
    # print(
    #     f"Entrada: {entradas[i]}, Saída da Rede: {round(camada_saida[0][0])}, Saída Esperada: {saidas[i][0]}")

# Visualização dos dados e da linha de decisão
# Selecionar duas características para visualização
# Ajustar esses índices para selecionar diferentes características
feature1, feature2 = 0, 1
plt.scatter(entradas[:, feature1], entradas[:, feature2],
            c=saidas, s=100, cmap='viridis', edgecolors='k')
plt.xlabel(f'Característica {feature1 + 1}')
plt.ylabel(f'Característica {feature2 + 1}')

# Calcular pontos da linha de decisão
x1_min, x1_max = entradas[:, feature1].min(
) - 0.5, entradas[:, feature1].max() + 0.5
x2_min, x2_max = entradas[:, feature2].min(
) - 0.5, entradas[:, feature2].max() + 0.5
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                       np.arange(x2_min, x2_max, 0.02))

# Função para aplicar a rede neural em uma entrada 2D


def apply_neural_network(x1, x2):
    input_data = np.zeros((1, 30))
    input_data[0, feature1] = x1
    input_data[0, feature2] = x2
    camada_oculta1 = relu(np.dot(input_data, pesos0))
    camada_oculta2 = relu(np.dot(camada_oculta1, pesos1))
    camada_oculta3 = relu(np.dot(camada_oculta2, pesos2))
    camada_saida = sigmoid(np.dot(camada_oculta3, pesos3))
    return camada_saida[0, 0]


# Aplicar a rede neural a cada ponto da grade
Z = np.array([apply_neural_network(x1, x2)
             for x1, x2 in zip(np.ravel(xx1), np.ravel(xx2))])
Z = Z.reshape(xx1.shape)

# Plotar a linha de decisão
plt.contourf(xx1, xx2, Z, alpha=0.4, cmap='coolwarm')
plt.colorbar()

# Adicionar legenda
maligno = plt.Line2D([0], [0], marker='o', color='w',
                     markerfacecolor='purple', markersize=10, label='Maligno')
benigno = plt.Line2D([0], [0], marker='o', color='w',
                     markerfacecolor='yellow', markersize=10, label='Benigno')
plt.legend(handles=[maligno, benigno])

# Destacar pontos de erro
predicted_classes = np.round([apply_neural_network(
    entrada[feature1], entrada[feature2]) for entrada in entradas])
incorrect = np.where(predicted_classes != saidas.flatten())[0]
plt.scatter(entradas[incorrect, feature1], entradas[incorrect, feature2],
            facecolors='none', edgecolors='r', s=100, label='Erro de Classificação')

plt.title('Dataset de Câncer de Mama - Linha de Decisão da Rede Neural')
plt.show()
