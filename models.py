import tensorflow as tf
from keras import layers, optimizers, Model
from typing import Dict, Any, Callable, Tuple

# --- Helper-Funktionen für Layer-Stacks ---

def _build_rnn_stack(input_tensor: tf.Tensor,
                     layer_type: Callable, # z.B. layers.LSTM oder layers.GRU
                     units: int,
                     n_layers: int,
                     bidirectional: bool = False,
                     layer_name_prefix: str = 'rnn') -> tf.Tensor:
    """Baut einen Stapel von RNN-Layern (LSTM oder GRU)."""
    x = input_tensor
    for i in range(n_layers):
        is_last_layer = (i == n_layers - 1)
        return_sequences = not is_last_layer
        rnn_layer = layer_type(units, return_sequences=return_sequences, name=f'{layer_name_prefix}_{i+1}')
        if bidirectional:
            # Name für Bidirectional Layer explizit setzen, um Konflikte zu vermeiden
            # (obwohl Keras dies oft automatisch gut handhabt)
            bidir_name = f'bidirectional_{layer_name_prefix}_{i+1}'
            x = layers.Bidirectional(rnn_layer, name=bidir_name)(x)
        else:
            x = rnn_layer(x)
    return x

def _build_conv_stack(input_tensor: tf.Tensor,
                        filters: int,
                        kernel_size: int,
                        n_layers: int,
                        activation: str = 'relu',
                        conv_type: str = 'cnn', # 'cnn' oder 'tcn'
                        increase_filters: bool = True, # Filter pro Layer erhöhen?
                        layer_name_prefix: str = 'conv1d') -> tf.Tensor:
    """
    Baut einen Stapel von Conv1D-Layern, unterstützt 'cnn' oder 'tcn'-Verhalten.
    ERSETZT _build_cnn_stack und _build_tcn_stack.

    Args:
        input_tensor: Der Eingabe-Tensor.
        filters: Anfangszahl der Filter für die erste Schicht.
        kernel_size: Größe des Faltungskerns.
        n_layers: Anzahl der Conv1D-Schichten.
        activation: Aktivierungsfunktion.
        conv_type: 'cnn' für Standard Conv1D (padding='same', dilation=1) oder
                   'tcn' für TCN-ähnlich (padding='causal', dilation=2**i).
        increase_filters: Wenn True, wird die Filteranzahl gemäß filters * (2**i)
                          für Layer i > 0 erhöht (wie im ursprünglichen Code).
        layer_name_prefix: Präfix für die Layer-Namen.

    Returns:
        Der Ausgabe-Tensor des Conv1D-Stapels.
    """
    x = input_tensor
    for i in range(n_layers):
        if conv_type == 'tcn':
            padding = 'causal'
            dilation_rate = 2**i
        elif conv_type == 'cnn':
            padding = 'same'
            dilation_rate = 1
        # Filterzahl für aktuellen Layer bestimmen
        layer_filters = filters * (2**i) if (increase_filters and i > 0) else filters
        conv_layer = layers.Conv1D(filters=layer_filters,
                                   kernel_size=kernel_size,
                                   activation=activation,
                                   padding=padding,
                                   dilation_rate=dilation_rate,
                                   name=f'{layer_name_prefix}_{conv_type}_{i+1}')
        x = conv_layer(x)
    return x

def _build_convlstm1d_stack(input_tensor: tf.Tensor,
                            filters: int,
                            kernel_size: int,
                            n_layers: int,
                            activation: str = 'tanh',
                            increase_filters: bool = False, # Filter pro Layer erhöhen?
                            layer_name_prefix: str = 'convlstm1d') -> tf.Tensor:
    """
    Baut einen Stapel von ConvLSTM1D-Layern, unterstützt 'convlstm' (standard)
    oder 'tcn'-Verhalten (causal padding, steigende dilation).
    """
    x = input_tensor
    current_filters = filters

    for i in range(n_layers):
        # Filterzahl erhöhen (optional, nach der ersten Schicht)
        if increase_filters and i > 0:
             #current_filters = filters * (2**i)
             current_filters *= 2

        is_last_layer = (i == n_layers - 1)
        return_sequences_flag = not is_last_layer if n_layers > 1 else False

        convlstm_layer = layers.ConvLSTM1D(
            filters=current_filters,
            kernel_size=kernel_size,
            activation=activation,
            padding='same',
            return_sequences=return_sequences_flag,
            name=f'{layer_name_prefix}_{i+1}' # Name angepasst
        )
        x = convlstm_layer(x)
        # Optional: Batch Normalization etc.
        # x = layers.BatchNormalization()(x)
    return x

# --- Einzelne Modell-Builder-Funktionen ---

def build_fnn(n_features: int, output_dim: int, hp: Dict[str, Any]) -> Model:
    """Baut ein Feedforward Neural Network (FNN)."""
    units = hp.get('units', 32)
    n_layers = hp.get('n_layers', 1) # Anzahl der *hidden* Dense Layer

    input_layer = layers.Input(shape=(output_dim, n_features), name='input')
    x = input_layer
    for i in range(n_layers):
         x = layers.Dense(units=units, activation='relu', name=f'dense_{i+1}')(x)
    x = layers.Flatten(name='flatten')(x)
    output_layer = layers.Dense(output_dim, name='output')(x)
    return Model(inputs=input_layer, outputs=output_layer, name='fnn')

def build_conv1d(n_features: int, output_dim: int, hp: Dict[str, Any], conv_type: str) -> Model:
    """
    Baut ein einfaches Conv1D-basiertes Modell (CNN oder TCN).
    ERSETZT build_cnn und build_tcn.

    Args:
        n_features: Anzahl der Eingabemerkmale.
        output_dim: Dimension der Ausgabe (Zeitschritte).
        hp: Dictionary mit Hyperparametern.
        conv_type: Der Typ der Faltung ('cnn' oder 'tcn').

    Returns:
        Das kompilierte Keras-Modell.
    """
    # Extrahiere Hyperparameter
    filters = hp.get('filters', 16)
    kernel_size = hp.get('kernel_size', 2)
    # Unterschiedliche Standardwerte für n_layers je nach Typ beibehalten?
    default_n_layers = 1 if conv_type == 'cnn' else 2
    n_layers = hp.get('n_cnn_layers', default_n_layers)
    increase_filters = hp.get('increase_filters', True)
    activation = hp.get('activation', 'relu') # Wird an Stack-Builder übergeben

    # Baue das Modell
    input_layer = layers.Input(shape=(output_dim, n_features), name='input')

    # Rufe den vereinheitlichten Stack-Builder auf
    conv_output = _build_conv_stack(
        input_tensor=input_layer,
        filters=filters,
        kernel_size=kernel_size,
        n_layers=n_layers,
        activation=activation, # Übergabe der Aktivierung
        conv_type=conv_type,   # Der entscheidende Parameter
        increase_filters=increase_filters,
        layer_name_prefix=conv_type # Nutze Typ als Prefix für Layer im Stack
    )

    x = layers.Flatten(name='flatten')(conv_output)
    output_layer = layers.Dense(output_dim, name='output')(x)

    # Nutze conv_type als Modellnamen
    model = Model(inputs=input_layer, outputs=output_layer, name=conv_type)
    return model

def build_rnn(n_features: int,
              output_dim: int,
              hp: Dict[str, Any],
              layer_type: Callable,
              bidirectional: bool = False,
              name: str = 'rnn') -> Model:
    """Baut ein Long Short-Term Memory (LSTM) Netzwerk."""
    units = hp.get('units', 16)
    n_layers = hp.get('n_rnn_layers', 1)
    input_layer = layers.Input(shape=(output_dim, n_features), name='input')
    x = _build_rnn_stack(input_layer, layer_type, units, n_layers, bidirectional=bidirectional, layer_name_prefix=name)
    output_layer = layers.Dense(output_dim, name='output')(x)
    return Model(inputs=input_layer, outputs=output_layer, name=name)

# --- Hybrid Modell-Builder ---

def build_cnn_rnn(n_features: int, output_dim: int, hp: Dict[str, Any],
                  conv_type: str, # 'cnn' oder 'tcn'
                  rnn_layer_type: Callable,
                  rnn_bidirectional: bool,
                  model_name: str) -> Model:
    """
    Generische Builder-Funktion für Conv1D-RNN Hybride (CNN oder TCN basiert).
    ERSETZT build_cnn_rnn und build_tcn_rnn.
    """
    # Hyperparameter extrahieren (Namen könnten vereinheitlicht werden, z.B. n_conv_layers statt n_cnn_layers)
    n_conv_layers = hp.get('n_cnn_layers', 1) # Versuch beide alten Namen zu lesen
    n_rnn_layers = hp.get('n_rnn_layers', 1)
    filters = hp.get('filters', 16)
    kernel_size = hp.get('kernel_size', 2)
    units = hp.get('units', 16)
    increase_filters = hp.get('increase_filters', True)
    conv_activation = hp.get('conv_activation', 'relu') # Aktivierung für Conv1D Stack

    # Baue das Modell
    input_layer = layers.Input(shape=(output_dim, n_features), name='input')

    conv_output = _build_conv_stack(
        input_tensor=input_layer,
        filters=filters,
        kernel_size=kernel_size,
        n_layers=n_conv_layers,
        activation=conv_activation,
        conv_type=conv_type, # Der entscheidende Parameter
        increase_filters=increase_filters,
        layer_name_prefix=conv_type # z.B. 'cnn' oder 'tcn' als Prefix
    )
    rnn_output = _build_rnn_stack(
        input_tensor=conv_output,
        layer_type=rnn_layer_type,
        units=units,
        n_layers=n_rnn_layers,
        bidirectional=rnn_bidirectional,
        layer_name_prefix=model_name.split('-')[-1] # z.B. 'lstm', 'gru', 'bilstm'
    )
    output_layer = layers.Dense(output_dim, name='output')(rnn_output)
    return Model(inputs=input_layer, outputs=output_layer, name=model_name)

def build_convlstm1d(n_features: int, output_dim: int, hp: Dict[str, Any]) -> Model:
    """
    Baut ein ConvLSTM1D Netzwerk unter Verwendung von _build_convlstm1d_stack.
    Der ConvLSTM1D-Stack kann als Standard ('convlstm') oder TCN-ähnlich ('tcn')
    konfiguriert werden.
    """
    # Hyperparameter extrahieren mit Defaults
    filters = hp.get('filters', 64)
    kernel_size = hp.get('kernel_size', 3)
    n_layers = hp.get('n_cnn_layers', 1)
    activation = hp.get('activation', 'tanh')
    increase_filters = hp.get('increase_filters', True) # Standardmäßig keine Erhöhung

    # --- Rest der Funktion bleibt fast gleich ---

    # Input Layer
    input_layer = layers.Input(shape=(output_dim, n_features), name='input')
    # Reshape für ConvLSTM1D
    reshaped_input = layers.Reshape((output_dim, 1, n_features),
                                  name='reshape_for_convlstm1d')(input_layer)
    convlstm_output = _build_convlstm1d_stack(
        input_tensor=reshaped_input,
        filters=filters,
        kernel_size=kernel_size,
        n_layers=n_layers,
        activation=activation,
        increase_filters=increase_filters,
        layer_name_prefix='convlstm1d'
    )
    x = layers.Flatten(name='flatten')(convlstm_output)
    output_layer = layers.Dense(output_dim, name='output')(x)
    model_name = f"convlstm1d"
    model = Model(inputs=input_layer, outputs=output_layer, name=model_name)
    return model

# --- Mappings für Dispatching ---

MODEL_BUILDERS = {
    'fnn': build_fnn,
    'cnn': lambda n, o, hp: build_conv1d(n, o, hp, conv_type='cnn'),
    'tcn': lambda n, o, hp: build_conv1d(n, o, hp, conv_type='tcn'),
    'lstm': lambda n, o, hp: build_rnn(n, o, hp, layers.LSTM, False, 'lstm'),
    'bilstm': lambda n, o, hp: build_rnn(n, o, hp, layers.LSTM, True, 'bilstm'),
    'gru': lambda n, o, hp: build_rnn(n, o, hp, layers.GRU, False, 'gru'),
    'bigru': lambda n, o, hp: build_rnn(n, o, hp, layers.GRU, True, 'bigru'),
    'convlstm': build_convlstm1d,
    # Hybride - nutzen Partials oder Lambdas, um Parameter an generische Builder zu übergeben
    'cnn-lstm': lambda n, o, hp: build_cnn_rnn(n, o, hp, 'cnn', layers.LSTM, False, 'cnn-lstm'),
    'cnn-gru': lambda n, o, hp: build_cnn_rnn(n, o, hp, 'cnn', layers.GRU, False, 'cnn-gru'),
    'cnn-bilstm': lambda n, o, hp: build_cnn_rnn(n, o, hp, 'cnn', layers.LSTM, True, 'cnn-bilstm'),
    'cnn-bigru': lambda n, o, hp: build_cnn_rnn(n, o, hp, 'cnn', layers.GRU, True, 'cnn-bigru'),
    'tcn-lstm': lambda n, o, hp: build_cnn_rnn(n, o, hp, 'tcn', layers.LSTM, False, 'tcn-lstm'),
    'tcn-gru': lambda n, o, hp: build_cnn_rnn(n, o, hp, 'tcn', layers.GRU, False, 'tcn-gru'),
    'tcn-bilstm': lambda n, o, hp: build_cnn_rnn(n, o, hp, 'tcn', layers.LSTM, True, 'tcn-bilstm'),
    'tcn-bigru': lambda n, o, hp: build_cnn_rnn(n, o, hp, 'tcn', layers.GRU, True, 'tcn-bigru'),
}

OPTIMIZERS = {
    'adam': optimizers.Adam,
    'rmsprop': optimizers.RMSprop,
    # Füge hier bei Bedarf weitere Optimizer hinzu
}

# --- Hauptfunktion zum Erstellen und Kompilieren des Modells ---

def get_metrics(config: dict) -> list:
    metric_list = config['model']['metrics']
    metrics = []
    for m in metric_list:
        if m == 'mae': metrics.append(tf.keras.metrics.MeanAbsoluteError(name='mae'))
        elif m == 'rmse': metrics.append(tf.keras.metrics.RootMeanSquaredError(name='rmse'))
        elif m == 'r^2': metrics.append(tf.keras.metrics.R2Score(name='r^2'))
    return metrics

def get_model(config: Dict[str, Any],
              n_features: int,
              output_dim: int,
              hyperparameters: Dict[str, Any]) -> Model:
    """
    Erstellt und kompiliert ein Keras-Modell basierend auf der Konfiguration
    und den Hyperparametern.
    """
    model_name = hyperparameters.get('model_name')
    if not model_name:
        raise ValueError("Hyperparameter 'model_name' ist erforderlich.")

    builder = MODEL_BUILDERS.get(model_name)
    if not builder:
        raise ValueError(f"Unbekannter Modellname: '{model_name}'. Verfügbar: {list(MODEL_BUILDERS.keys())}")

    # Baue das Modell
    model = builder(n_features, output_dim, hyperparameters)
    # model.summary() # Optional: Modellzusammenfassung ausgeben

    # Wähle den Optimizer
    optimizer_name = config.get('model', {}).get('optimizer', 'adam') # Default zu Adam
    learning_rate = hyperparameters.get('lr', 0.001) # Default Learning Rate
    optimizer_class = OPTIMIZERS.get(optimizer_name)
    if not optimizer_class:
        raise ValueError(f"Unbekannter Optimizer: '{optimizer_name}'. Verfügbar: {list(OPTIMIZERS.keys())}")

    optimizer = optimizer_class(learning_rate=learning_rate)

    # Kompiliere das Modell
    loss = config.get('model', {}).get('loss', 'mse') # Default zu MSE
    metrics = config.get('model', {}).get('metrics', ['mae']) # Default zu MAE
    metrics = get_metrics(config=config)
    # Sicherstellen, dass Metriken eine Liste ist, falls nur ein String übergeben wird
    if isinstance(metrics, str):
        metrics = [metrics]

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model
