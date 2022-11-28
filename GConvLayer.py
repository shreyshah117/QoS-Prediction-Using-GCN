class GConv(Layer):
    def __init__(self, adj, units=32, activation=None):
        super(GConv, self).__init__()
        self.adj = adj
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.w1 = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.w2 = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b1 = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )
        self.b2 = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )
        
    def call(self, inputs):   
        embedding = self.activation(self.adj @ inputs @ self.w1 + self.b1) #graph conv
        #embedding = self.activation(inputs @ self.w1 + self.b1)
        mlp_embedding = self.activation(embedding @ self.w2 + self.b2)                      #dense mlp
        #mlp_embedding = tf.nn.dropout(mlp_embedding, 1 - self.mess_dropout[k]) #dropout
        return mlp_embedding

