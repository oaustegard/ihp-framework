import tensorflow as tf
from tensorflow.keras import layers, Model

class DilatedCausalConv1D(layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate):
        super(DilatedCausalConv1D, self).__init__()
        self.conv = layers.Conv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  padding='causal',
                                  dilation_rate=dilation_rate)
    
    def call(self, inputs):
        return self.conv(inputs)

class MTCNLayer(layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate):
        super(MTCNLayer, self).__init__()
        self.conv = DilatedCausalConv1D(filters, kernel_size, dilation_rate)
        self.norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout_rate)
        self.activation = layers.Activation('relu')
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        x = self.activation(x)
        return self.dropout(x)

class MTCN(Model):
    def __init__(self, num_layers, kernel_size, filters, dropout_rate, output_dim):
        super(MTCN, self).__init__()
        self.layers_list = [MTCNLayer(filters, kernel_size, 2**i, dropout_rate) 
                            for i in range(num_layers)]
        self.final_conv = layers.Conv1D(filters=output_dim, kernel_size=1)
    
    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return self.final_conv(x)

def build_mtcn(input_shape, num_layers, kernel_size, filters, dropout_rate, output_dim):
    inputs = layers.Input(shape=input_shape)
    mtcn = MTCN(num_layers, kernel_size, filters, dropout_rate, output_dim)
    outputs = mtcn(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    return model

class MTCNPredictor:
    def __init__(self, input_shape, num_layers, kernel_size, filters, dropout_rate, output_dim):
        self.model = build_mtcn(input_shape, num_layers, kernel_size, filters, dropout_rate, output_dim)
    
    def compile(self, optimizer='adam', loss='mse'):
        self.model.compile(optimizer=optimizer, loss=loss)
    
    def fit(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, filepath):
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath):
        loaded_model = tf.keras.models.load_model(filepath, custom_objects={
            'DilatedCausalConv1D': DilatedCausalConv1D,
            'MTCNLayer': MTCNLayer,
            'MTCN': MTCN
        })
        predictor = cls(loaded_model.input_shape[1:], 0, 0, 0, 0, 0)  # Dummy values
        predictor.model = loaded_model
        return predictor

# Usage example:
# input_shape = (100, 10)  # 100 time steps, 10 features
# num_layers = 6
# kernel_size = 3
# filters = 64
# dropout_rate = 0.1
# output_dim = 1
# 
# predictor = MTCNPredictor(input_shape, num_layers, kernel_size, filters, dropout_rate, output_dim)
# predictor.compile()
# predictor.fit(X_train, y_train)
# predictions = predictor.predict(X_test)
