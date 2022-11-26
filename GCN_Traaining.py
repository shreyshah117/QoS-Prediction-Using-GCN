##### training our gcn
import time 

n_users = 339
n_servs = 5825

#F = sf.shape[1]
#F = encoder_stat.shape[1]
F = node_feature.shape[1]
#F = encoder_f.shape[1]
#F = svd_f.shape[1]
#F = svd_stat.shape[1]


class GCNBlock(Layer):
    def __init__(self, adj_hat, units):
        super(GCNBlock, self).__init__()
        self.adj_hat = adj_hat
        self.units = units
        self.conv1 = GConv(self.adj_hat, units = self.units, activation = 'relu')
        self.conv2 = GConv(self.adj_hat, units = self.units, activation = 'relu')
        self.conv3 = GConv(self.adj_hat, units = self.units, activation = 'relu')
        self.bilinear = Bilinear(units = 3*self.units, activation = 'relu')        
    
    def call(self, inputs):
        x1 = self.conv1(inputs)
        
        x2 = self.conv2(x1)
        
        x3 = self.conv3(x2)
        
        #print(inputs.shape, x1.shape, x2.shape, x3.shape )
    
        #all_embeddings = tf.reduce_sum([inputs, x1 , x2, x3], 0)
        all_embeddings = tf.concat([x1 , x2, x3], 1)
        #all_embeddings = tf.concat([x1 , x2], 1)
        #all_embeddings = x1
          
        
        #print(all_embeddings.shape)
        
        u_all_embed, s_all_embed = tf.split(all_embeddings, [n_users, n_servs], 0)
        
        #print(u_all_embed.shape, s_all_embed.shape)
        
        pred_rt = self.bilinear(u_all_embed, s_all_embed)
        
        #pred_rt = tf.matmul(u_all_embed, s_all_embed, transpose_a=False, transpose_b=True, name='output')
        
        return pred_rt


#input_f = tf.cast(svd_stat, dtype='float32')
#input_f = tf.cast(encoder_f, dtype='float32')
#input_f = tf.cast(svd_f, dtype='float32') 
#input_f = tf.cast(sf, dtype='float32')
#input_f = tf.cast(encoder_stat, dtype='float32')
input_f = tf.cast(node_feature, dtype='float32')

adj_normalized = tf.cast(adj_normalized, dtype='float32')

gcn = GCNBlock(adj_normalized, F)

inputs = Input(shape=(F,), name='node_feature')
outputs = gcn(inputs)

model = Model(inputs, outputs)


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = Custom_Loss()


#The early stopping strategy: stop the training if `val_loss` does not
    # decrease over a certain number of epochs.
patience = 1000
wait = 0
best = 1
    
@tf.function
def train():
    with tf.GradientTape() as tape:
        pred = model(input_f, training = True)
        loss_value = loss_fn(rtdata_10, pred)
        loss_value += sum(model.losses)      
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))  
    return loss_value


for epoch in range(10000):
    start_time = time.time()
    loss_value = train()
    print(f"epoch: {epoch:d} -- loss: {loss_value:.3f} -- time-taken:{time.time()-start_time} ") 
    wait += 1
    if loss_value < best:
        best = loss_value
        wait = 0
    if wait >= patience:
        print("**********Early Training END**********")
        break

        
        
rt_pred = model(input_f, training=False)
rt_pred = rt_pred.numpy()

count = 0
ae=0

for i in range(339):
    for j in range(5825):
        if rtdata_O[i][j]!=0 and rtdata_10[i][j]==0:
            ae += np.abs(rt_pred[i][j]-rtdata_O[i][j])
            count +=1
            
print('Error = {0:.3f}'.format(ae/count))
