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

"""
class Bilinear(Layer):
    def __init__(self, units=32, activation=None):
        super(Bilinear, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):  # Create the state of the layer (weights)
        w_init = tf.random_normal_initializer()
        self.w1 = tf.Variable(
            initial_value=w_init(shape=(self.units, self.units),
                                 dtype='float32'),
            trainable=True)
        
        self.w2 = tf.Variable(
            initial_value=w_init(shape=(self.units, self.units),
                                 dtype='float32'),
            trainable=True)
        
        self.w3 = tf.Variable(
            initial_value=w_init(shape=(self.units, self.units),
                                 dtype='float32'),
            trainable=True)
        
        b_init = tf.zeros_initializer()      
        self.b1 = tf.Variable(
            initial_value=b_init(shape=(self.units,), dtype='float32'),
            trainable=True)
        
        self.b2 = tf.Variable(
            initial_value=b_init(shape=(self.units,), dtype='float32'),
            trainable=True)
        
        self.b3 = tf.Variable(
            initial_value=b_init(shape=(self.units,), dtype='float32'),
            trainable=True)

    def call(self, input1, input2):  # Defines the computation from inputs to outputs
        
        output1 = self.activation(tf.matmul(input1, self.w1) + self.b1)
        output2 = self.activation(tf.matmul(input2, self.w2) + self.b2)
        
        
        output3 = tf.matmul(tf.matmul(output1, self.w3), tf.transpose(output2))
        ## u*w*s, output3 size 339*5825
        print(output3.shape)
    
        return output3
 """
