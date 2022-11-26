from tensorflow.keras import Sequential, Model
class User_AutoEncoders(Model):
    def __init__(self, output_units):
        super().__init__()
        self.encoder = Sequential([
              Dense(500, activation="relu"),
              Dense(250, activation="relu"),
              Dense(100, activation="relu")])

        self.decoder = Sequential([
              Dense(250, activation="relu"),
              Dense(500, activation="relu"),
              Dense(output_units, activation="linear")])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
 class Serv_AutoEncoders(Model):
    def __init__(self, output_units):
        super().__init__()
        self.encoder = Sequential([
              Dense(3000, activation="relu"),
              Dense(750, activation="relu"),
              Dense(100, activation="relu")])

        self.decoder = Sequential([
              Dense(750, activation="relu"),
              Dense(3000, activation="relu"),
              Dense(output_units, activation="linear")])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

serv_auto_encoder = Serv_AutoEncoders(len(s_sp_wsdl_csm_hd[1]))

serv_auto_encoder.compile(loss='mse', metrics=['mae'], optimizer='adam')
callback_serv = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

serv_history = serv_auto_encoder.fit(s_sp_wsdl_csm_hd, s_sp_wsdl_csm_hd, 
    epochs=200, 
    batch_size=128, callbacks=[callback_serv],
    validation_split=0.2)

user_auto_encoder = User_AutoEncoders(len(uc_as_csm_hd[1]))

user_auto_encoder.compile(loss='mse', metrics=['mae'], optimizer='adam')
callback_user = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

user_history = user_auto_encoder.fit(uc_as_csm_hd, uc_as_csm_hd, 
    epochs=500, 
    batch_size=128, callbacks=[callback_user],
    validation_split=0.2)


