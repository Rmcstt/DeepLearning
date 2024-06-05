import tensorflow as tf

# Definição da arquitetura da rede neural
rede = tf.keras.Sequential([
    # Camada de entrada com 2 neurônios
    tf.keras.layers.Input(shape=(2,)),
    # Camada oculta com 3 neurônios e função de ativação sigmoid
    tf.keras.layers.Dense(3, activation='sigmoid'),
    # Camada de saída com 1 neurônio e função de ativação sigmoid
    tf.keras.layers.Dense(3, activation='sigmoid')
])

# Compilação do modelo
rede.compile(optimizer='adam', loss='binary_crossentropy',
             metrics=['accuracy'])

# Sumário da arquitetura da rede
rede.summary()
