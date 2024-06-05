import numpy as np

# sigmoid


def sigmoid(soma):
    return 1 / (1 + np.exp(- soma))


a = np.exp(-7)
b = sigmoid(-1)

print(a)
print(b)
