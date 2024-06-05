from sklearn.neural_network import MLPClassifier
from sklearn import datasets

iris = datasets.load_iris()
entradas = iris.data
saidas = iris.target

redeNeural = MLPClassifier(verbose=True,
                           max_iter=1000,
                           tol=0.00001,
                           activation='logistic',
                           learning_rate_init=0.001)

redeNeural.fit(entradas, saidas)
print(iris.target_names[redeNeural.predict([[7, 7.2, 5.9, 5.2]])])
