import tensorflow as tf
from keras import layers, optimizers, Model

def get_model(config: dict,
              n_features: int,
              output_dim: int,
              hyperparameters: dict):
    if hyperparameters['model_name'] == 'fnn':
        model = fnn(n_features,
                    output_dim,
                    hyperparameters['n_layers'], # n_layer
                    hyperparameters['units']) # units

    if hyperparameters['model_name'] == 'cnn':
        model = cnn(n_features,
                    output_dim,
                    hyperparameters['n_layers'], # n_layer
                    hyperparameters['filters'], # filters
                    hyperparameters['kernel_size']) # kernel_size

    if hyperparameters['model_name'] == 'cnn-lstm':
        model = cnn_lstm(n_features,
                    output_dim,
                    hyperparameters['n_cnn_layers'],
                    hyperparameters['n_rnn_layers'],
                    hyperparameters['filters'], # filters
                    hyperparameters['kernel_size'],
                    hyperparameters['units']) # kernel_size

    elif hyperparameters['model_name'] == 'tcn':
        model = tcn(n_features,
                    output_dim,
                    hyperparameters['n_layers'], # n_layer
                    hyperparameters['filters'], # filters
                    hyperparameters['kernel_size']) # kernel_size

    elif hyperparameters['model_name'] == 'lstm':
        model = lstm(n_features,
                     output_dim,
                     hyperparameters['n_layers'], # n_layer
                     hyperparameters['units']) # units

    elif hyperparameters['model_name'] == 'bilstm':
        model = bilstm(n_features,
                       output_dim,
                       hyperparameters['n_layers'], # n_layer
                       hyperparameters['units']) # units

    elif hyperparameters['model_name'] == 'gru':
        model = gru(n_features,
                     output_dim,
                     hyperparameters['n_layers'], # n_layer
                     hyperparameters['units']) # units

    elif hyperparameters['model_name'] == 'bigru':
        model = bigru(n_features,
                       output_dim,
                       hyperparameters['n_layers'], # n_layer
                       hyperparameters['units'])
    if config['model']['optimizer'] == 'adam':
        optimizer = optimizers.Adam(learning_rate=hyperparameters['lr'])
    elif config['model']['optimizer'] == 'rmsprop':
        optimizer = optimizers.RMSprop(learning_rate=hyperparameters['lr'])
    model.compile(optimizer=optimizer,
                  loss=config['model']['loss'],
                  metrics=[config['model']['metrics']])
    return model

def fnn(n_features,
        output_dim,
        n_layers=1,
        units=32):
    input = layers.Input(shape=(output_dim, n_features))
    x = layers.Dense(units=units, activation='relu')(input)
    for layer in range(1, n_layers):
        x = layers.Dense(units=units, activation='relu')(x)
    x = layers.Flatten(name='flatten_time')(x)
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

def gru(n_features,
         output_dim,
         n_layers=1,
         units=16):
    input = layers.Input(shape=(output_dim, n_features))
    if n_layers == 1:
        x = layers.GRU(units)(input)
    else:
        x = layers.GRU(units, return_sequences=True)(input)
        for layer in range(1, n_layers):
            if layer == (n_layers-1):
                x = layers.GRU(units)(x)
            else:
                x = layers.GRU(units, return_sequences=True)(x)
    output = layers.Dense(output_dim, name='output')(x)
    model = Model(inputs=input, outputs=output)
    return model

def bigru(n_features,
           output_dim,
           n_layers=1,
           units=16):
    input = layers.Input(shape=(output_dim, n_features))
    if n_layers == 1:
        x = layers.Bidirectional(layers.GRU(units))(input)
    else:
        x = layers.Bidirectional(layers.GRU(units, return_sequences=True))(input)
        for layer in range(1, n_layers):
            if layer == (n_layers-1):
                x = layers.Bidirectional(layers.GRU(units))(x)
            else:
                x = layers.Bidirectional(layers.GRU(units, return_sequences=True))(x)
    output = layers.Dense(output_dim, name='output')(x)
    model = Model(inputs=input, outputs=output)
    return model

def cnn_lstm(n_features,
             output_dim,
             n_cnn_layers=1,
             n_lstm_layers=1,
             filters=16,
             kernel_size=2,
             units=16):
    input = layers.Input(shape=(output_dim, n_features))
    x = layers.Conv1D(filters=filters,
                      kernel_size=kernel_size,
                      activation='relu',
                      padding='same')(input)
    if n_cnn_layers > 1:
        for layer in range(1, n_cnn_layers):
            current_filters = filters * (2**layer)
            x = layers.Conv1D(filters=current_filters,
                              kernel_size=kernel_size,
                              activation='relu',
                              padding='same')(x)
    # Die Ausgabe des CNN-Teils hat die Form (batch, output_dim, filters)
    # und kann direkt an das LSTM übergeben werden.
    if n_lstm_layers == 1:
        x = layers.LSTM(units)(x)
    else:
        x = layers.LSTM(units, return_sequences=True)(x)
        for layer in range(1, n_lstm_layers):
            if layer == (n_lstm_layers - 1):
                x = layers.LSTM(units)(x)
            else:
                x = layers.LSTM(units, return_sequences=True)(x)
    output = layers.Dense(output_dim, name='output')(x)
    model = Model(inputs=input, outputs=output)
    return model
