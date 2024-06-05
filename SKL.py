from sklearn.neural_network import MLPClassifier
from sklearn import datasets

iris = datasets.load_iris()

entradas = iris.data
saidas = iris.target

redeneural = MLPClassifier(verbose=True,
                           max_iter=100000,
                           tol=0.00001,
                           activation='logistic',
                           learning_rate_init=0.00001)
redeneural.fit(entradas, saidas)
