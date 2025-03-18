import tensorflow as tf
from keras import layers, optimizers, Model


def fnn(n_features,
        output_dim,
        n_layers=1,
        units=32):
    input = layers.Input(shape=(output_dim, n_features))
    
    x = layers.Dense(units=units, activation='relu')(input)
    
    for layer in range(1, n_layers):
        x = layers.Dense(units=units, activation='relu')(x)

    output = layers.Dense(output_dim, name='output')(x)
    
    model = Model(inputs=input, outputs=output)
    return model


def cnn(n_features, 
        output_dim, 
        n_layers=1, 
        filters=16, 
        kernel_size=2):

    input = layers.Input(shape=(output_dim, n_features))
    x = layers.Conv1D(filters=filters, 
                      kernel_size=kernel_size, 
                      activation='relu', 
                      padding='same')(input)
    if n_layers > 1:
        for layer in range(1, n_layers):
            x = layers.Conv1D(filters=filters*2**layer, 
                              kernel_size=kernel_size, 
                              activation='relu', 
                              padding='same')(x)
    x = layers.Flatten()(x)
    output = layers.Dense(output_dim, name='output')(x)
    
    model = Model(inputs=input, outputs=output)
    return model


def tcn(n_features, 
        output_dim, 
        n_layers=2, 
        filters=16, 
        kernel_size=2):

    input = layers.Input(shape=(output_dim, n_features))
    x = layers.Conv1D(filters=filters, 
                      kernel_size=kernel_size, 
                      activation='relu', 
                      padding='causal', 
                      dilation_rate=1)(input)
    for layer in range(1, n_layers):
        x = layers.Conv1D(filters=filters*2**layer, 
                            kernel_size=kernel_size, 
                            activation='relu', 
                            padding='causal', 
                            dilation_rate=2**layer)(x)
    x = layers.Flatten()(x)
    output = layers.Dense(output_dim, name='output')(x)
    
    model = Model(inputs=input, outputs=output)
    return model


def lstm(n_features, 
         output_dim, 
         n_layers=1,
         units=16):
    
    input = layers.Input(shape=(output_dim, n_features))

    if n_layers == 1:
        x = layers.LSTM(units)(input)
    else:
        x = layers.LSTM(units, return_sequences=True)(input)
        for layer in range(1, n_layers):
            if layer == (n_layers-1):
                x = layers.LSTM(units)(x)
            else:
                x = layers.LSTM(units, return_sequences=True)(x)
    
    output = layers.Dense(output_dim, name='output')(x)

    model = Model(inputs=input, outputs=output)
    return model


def bilstm(n_features, 
           output_dim, 
           n_layers=1,
           units=16):
    
    input = layers.Input(shape=(output_dim, n_features))

    if n_layers == 1:
        x = layers.Bidirectional(layers.LSTM(units))(input)
    else:
        x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(input)
        for layer in range(1, n_layers):
            if layer == (n_layers-1):
                x = layers.Bidirectional(layers.LSTM(units))(x)
            else:
                x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    
    output = layers.Dense(output_dim, name='output')(x)

    model = Model(inputs=input, outputs=output)
    return model