import tensorflow as tf
import keras.ops
from keras import layers, optimizers, Model
from typing import Dict, Any, Callable, Tuple
import logging


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
            bidir_name = f'bi{layer_name_prefix}_{i+1}'
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
                        increase_filters: bool = False, # Filter pro Layer erhöhen?
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
                                   name=f'{layer_name_prefix}_{i+1}')
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

def _grn(x, hidden_dim, context=None, dropout_rate=0.1, name=''):
    """Gated Residual Network"""
    # Hole Input-Dimension aus der letzten Achse
    # tf.keras.backend.int_shape ist robuster für symbolische Tensoren
    # input_dim = tf.keras.backend.int_shape(x)[-1]
    # Oder über .shape Attribut, was für funktionale API üblich ist
    if x.shape[-1] is None:
         raise ValueError(f"GRN '{name}': Letzte Dimension von Input x darf nicht None sein. Shape: {x.shape}")
    input_dim = x.shape[-1]

    # Kontext hinzufügen, falls vorhanden
    if context is not None:
        # Stelle sicher, dass Kontext über Zeitdimension verteilt ist, falls x zeitlich ist
        # Prüfe Ränge (Anzahl Dimensionen)
        if len(x.shape) == 3 and len(context.shape) == 2: # x is (batch, time, features), context is (batch, features)
             # KORREKTUR: Verwende keras.ops.shape für dynamische Dimension
             time_steps = keras.ops.shape(x)[1]
             context_repeated = layers.RepeatVector(time_steps)(context) # Wiederhole Kontext für jeden Zeitschritt
        elif len(x.shape) == len(context.shape):
             # Wenn Ränge gleich sind, nehme an, Kontext ist bereits passend (z.B. (batch, time, context_features))
             context_repeated = context
        else:
             raise ValueError(f"GRN '{name}': Ränge von Input x ({len(x.shape)}) und Context ({len(context.shape)}) sind inkompatibel für Wiederholung/Projektion.")

        # Kontext auf passende Dimension projizieren
        # Sicherer ist eine Projektion, statt Annahmen über die Dimension zu treffen
        context_proj_dim = hidden_dim # Oder eine andere sinnvolle Größe, z.B. input_dim
        context_proj = layers.Dense(context_proj_dim, name=f'{name}_context_proj')(context_repeated)

        # Konkateniere Input und projizierten Kontext
        input_combined = layers.Concatenate(axis=-1)([x, context_proj])
        # Die Dimension nach Konkatenation (nur für interne Logik/Debug relevant)
        # dense_input_dim = input_dim + context_proj_dim
    else:
        # Kein Kontext vorhanden
        input_combined = x
        # dense_input_dim = input_dim

    # Zwischenschicht mit ELU-Aktivierung
    hidden_layer = layers.Dense(hidden_dim, activation='elu', name=f'{name}_dense1')(input_combined)
    # Dropout anwenden
    hidden_layer = layers.Dropout(dropout_rate, name=f'{name}_dropout1')(hidden_layer) # Dropout nach erster Dense+Activation
    # Zweite Dense-Schicht (oft linear oder mit anderer Aktivierung)
    hidden_layer = layers.Dense(hidden_dim, name=f'{name}_dense2')(hidden_layer)
    # Dropout optional auch hier anwendbar

    # Gating-Schicht
    gate = layers.Dense(hidden_dim, activation='sigmoid', name=f'{name}_gate')(input_combined) # Input ist die *kombinierte* Eingabe

    # Residual Connection
    # Wenn Input-Dimension != Output-Dimension (hidden_dim), lineares Mapping für Residual Connection
    if input_dim != hidden_dim:
        x_proj = layers.Dense(hidden_dim, name=f'{name}_input_proj')(x) # Projiziere Original-Input x
    else:
        x_proj = x # Keine Projektion nötig

    # Gated Output + Residual Connection + Layer Normalization
    gated_output = layers.Multiply()([hidden_layer, gate]) # Elementweise Multiplikation
    output = layers.Add()([x_proj, gated_output]) # Addiere (projizierten) Input
    output = layers.LayerNormalization(name=f'{name}_layernorm')(output) # Wende LayerNorm an

    return output

# Helper Function: Variable Selection Network (VSN)
def _variable_selection_network(x, num_features, hidden_dim, static_context, dropout_rate, name_prefix):
    """
    Variable Selection Network using GRN.
    Weights input features based on static context (if provided).
    Returns features transformed to hidden_dim.
    """
    repeated_context = None
    if static_context is not None:
        # FIX: Verwende keras.ops.shape für symbolische Tensoren
        time_steps = keras.ops.shape(x)[1]
        repeated_context = layers.RepeatVector(time_steps)(static_context)

    if num_features == 1:
        # Keine Variablenselektion nötig/möglich. Transformiere direkt.
        # Verwende GRN oder einfache Dense Schicht
        transformed_features = _grn(x, hidden_dim, context=repeated_context, dropout_rate=dropout_rate, name=f'{name_prefix}_single_feature_grn')
        # transformed_features = layers.TimeDistributed(layers.Dense(hidden_dim), name=f'{name_prefix}_single_feature_transform')(x)

        # Gib Dummy-Gewichte zurück (z.B. Tensor von Einsen oder None)
        # --- KORREKTUR HIER: tf.ones_like verwenden ---
        # ones_shape = keras.ops.shape(x) # Nicht mehr benötigt
        # feature_weights = tf.ones(ones_shape, dtype=x.dtype) # Fehlerhafte Zeile
        feature_weights = keras.ops.ones_like(x, dtype=x.dtype) # Erzeugt Tensor von Einsen mit gleicher Shape wie x
        # oder einfach: feature_weights = tf.ones_like(x) # dtype wird meist automatisch übernommen
        # --- Ende Korrektur ---

        return transformed_features, feature_weights

    # 1. GRN zur Erzeugung von Feature-Gewichten (Output-Dimension = num_features)
    #    Input für dieses GRN sind die Original-Features 'x' und der (optionale) Kontext.
    weight_grn_output = _grn(x, num_features, context=repeated_context, dropout_rate=dropout_rate, name=f'{name_prefix}_weight_grn')

    # 2. Softmax über die Feature-Dimension, um normalisierte Gewichte zu erhalten
    feature_weights = layers.TimeDistributed(layers.Softmax(axis=-1), name=f'{name_prefix}_softmax')(weight_grn_output)

    # Implementierungs-Variante: Gewichte berechnen -> Gewichte auf Originalfeatures anwenden -> Ergebnis transformieren
    # Wende Gewichte auf Original-Features an
    weighted_features = layers.Multiply(name=f'{name_prefix}_multiply')([x, feature_weights])

    # Transformiere die gewichteten Features auf hidden_dim
    # Wichtig: Kontext hier optional weitergeben, falls gewünscht/sinnvoll.
    # Hier verwenden wir keinen Kontext für die finale Transformation.
    transformed_weighted_features = layers.TimeDistributed(layers.Dense(hidden_dim), name=f'{name_prefix}_transform')(weighted_features)
    # ODER mit GRN (würde repeated_context benötigen, falls static_context vorhanden):
    # transformed_weighted_features = _grn(weighted_features, hidden_dim, context=repeated_context, dropout_rate=dropout_rate, name=f'{name_prefix}_final_grn')

    return transformed_weighted_features, feature_weights

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
    increase_filters = hp.get('increase_filters', False)
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
    increase_filters = hp.get('increase_filters', False)
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

# Special model builders

def build_convlstm1d(n_features: int,
                     output_dim: int,
                     hp: Dict[str, Any]) -> Model:
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
    increase_filters = hp.get('increase_filters', False) # Standardmäßig keine Erhöhung

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

def build_tft(feature_dims: Dict[str, int],
              output_dim: int,
              hyperparameters: Dict[str, Any]):
    """
    Erstellt ein Temporal Fusion Transformer Modell mit Variable Selection,
    Causal Mask und optionalen statischen Inputs.
    """
    # initialize feature_dims
    observed_dim = feature_dims['observed_dim']
    known_dim = feature_dims['known_dim']
    static_dim = feature_dims['static_dim']
    # initialize hyperparameters
    n_heads = hyperparameters['n_heads']
    sequence_length = hyperparameters['lookback']
    forecast_horizon = hyperparameters['horizon']
    hidden_dim = hyperparameters['hidden_dim']
    dropout_rate = hyperparameters['dropout']
    # --- Input Layer ---
    observed_input = layers.Input(shape=(sequence_length, observed_dim), name='observed_input')
    known_input = layers.Input(shape=(sequence_length + forecast_horizon, known_dim), name='known_input')

    model_inputs = [observed_input, known_input] # Start mit zeitlichen Inputs

    # --- Statische Verarbeitung (Optional) ---
    static_input = None
    static_context_vsn = None
    static_context_enrichment = None
    static_context_state_h = None
    static_context_state_c = None

    if static_dim > 0:
        static_input = layers.Input(shape=(static_dim,), name='static_input')
        model_inputs.append(static_input) # Füge Static Input hinzu

        # Erzeuge Kontexte nur, wenn statische Features vorhanden sind
        static_context_vsn = _grn(static_input, hidden_dim, dropout_rate=dropout_rate, name='static_grn_vsn')
        static_context_enrichment = _grn(static_input, hidden_dim, dropout_rate=dropout_rate, name='static_grn_enrichment')
        static_context_state_h = _grn(static_input, hidden_dim, dropout_rate=dropout_rate, name='static_grn_state_h')
        static_context_state_c = _grn(static_input, hidden_dim, dropout_rate=dropout_rate, name='static_grn_state_c')
    else:
        # Optional: Log oder Warnung ausgeben
        logging.info("TFT-Modell wird ohne statische Features erstellt.")
        pass # Kontexte bleiben None

    # --- Zeitvariable Verarbeitung MIT VARIABLE SELECTION ---
    # Trenne bekannte Inputs
    known_encoder_input = known_input[:, :sequence_length, :]
    known_decoder_input = known_input[:, sequence_length:, :]

    # Wende Variable Selection an (übergibt optionalen static_context_vsn)
    observed_selected, obs_weights = _variable_selection_network(
        observed_input, observed_dim, hidden_dim, static_context_vsn, dropout_rate, 'observed_vsn'
    )
    known_encoder_selected, known_enc_weights = _variable_selection_network(
        known_encoder_input, known_dim, hidden_dim, static_context_vsn, dropout_rate, 'known_encoder_vsn'
    )
    known_decoder_selected, known_dec_weights = _variable_selection_network(
        known_decoder_input, known_dim, hidden_dim, static_context_vsn, dropout_rate, 'known_decoder_vsn'
    )
    # --- LSTM Encoder ---
    encoder_input_combined = layers.Concatenate(axis=-1)([observed_selected, known_encoder_selected])
    lstm_encoder = layers.LSTM(hidden_dim,
                               return_sequences=True,
                               return_state=True,
                               name='lstm_encoder')
    # Initialisiere LSTM-Zustand nur, wenn Kontexte vorhanden sind
    initial_state_encoder = [static_context_state_h, static_context_state_c] if static_context_state_h is not None else None
    encoder_outputs, state_h, state_c = lstm_encoder(encoder_input_combined, initial_state=initial_state_encoder)

    # --- LSTM Decoder ---
    lstm_decoder = layers.LSTM(hidden_dim,
                               return_sequences=True,
                               name='lstm_decoder')
    # Initialisiere mit dem finalen Zustand des Encoders (immer vorhanden)
    initial_state_decoder = [state_h, state_c]
    decoder_outputs = lstm_decoder(known_decoder_selected, initial_state=initial_state_decoder)

    # --- Static Enrichment (Optional) ---
    lstm_outputs_combined = layers.Concatenate(axis=1)([encoder_outputs, decoder_outputs])
    enriched_temporal_features = lstm_outputs_combined # Standardwert

    if static_context_enrichment is not None:
        total_seq_len = sequence_length + forecast_horizon
        static_enrichment_repeated = layers.RepeatVector(total_seq_len)(static_context_enrichment)
        # Verwende GRN für Anreicherung
        enriched_temporal_features = _grn(lstm_outputs_combined, hidden_dim,
                                          context=static_enrichment_repeated,
                                          dropout_rate=dropout_rate, name='enrichment_grn')
    # else: Wenn kein statischer Kontext, wird lstm_outputs_combined direkt verwendet


    # --- Temporal Self-Attention MIT CAUSAL MASK ---
    # (Masken-Erzeugung bleibt gleich)
    seq_len_total = sequence_length + forecast_horizon
    i = tf.range(seq_len_total)[:, tf.newaxis]
    j = tf.range(seq_len_total)[tf.newaxis, :]
    causal_mask = j > i

    attention_output = layers.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=hidden_dim // n_heads,
        dropout=dropout_rate,
        name='multihead_attention'
    )(
        query=enriched_temporal_features, # Verwende angereicherte Features
        value=enriched_temporal_features,
        key=enriched_temporal_features,
        attention_mask=causal_mask
    )
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    # Residual Connection zum Input der Attention anwenden
    attention_output = layers.LayerNormalization()(enriched_temporal_features + attention_output)

    # --- Position-wise Feed-Forward ---
    # Input ist der Output der Add & Norm Schicht nach Attention
    ff_output = _grn(attention_output, hidden_dim, dropout_rate=dropout_rate, name='ff_grn')
    # Residual Connection zum Input des FF GRN
    ff_output = layers.LayerNormalization()(attention_output + ff_output)

    # --- Final Output Layer ---
    forecast_output_features = ff_output[:, sequence_length:, :]
    output = layers.TimeDistributed(layers.Dense(1), name='output_dense')(forecast_output_features)

    # --- Model Definition ---
    # Inputs hängen davon ab, ob static_input erstellt wurde
    model = Model(inputs=model_inputs,
                  outputs=output,
                  name='TemporalFusionTransformer_VSN_Masked_OptStatic')
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
    # Special models
    'convlstm': build_convlstm1d,
    'tft': build_tft,
    # Hybrid models
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
              hyperparameters: Dict[str, Any]) -> Model:
    """
    Erstellt und kompiliert ein Keras-Modell basierend auf der Konfiguration
    und den Hyperparametern.
    """
    model_name = config.get('model').get('name')
    if not model_name:
        raise ValueError("Hyperparameter 'model_name' erforderlich.")

    builder = MODEL_BUILDERS.get(model_name)
    if not builder:
        raise ValueError(f"Unbekannter Modellname: '{model_name}'. Verfügbar: {list(MODEL_BUILDERS.keys())}")

    output_dim = config['model']['output_dim']
    feature_dim = config['model']['feature_dim']
    model = builder(feature_dim, output_dim, hyperparameters)

    # optimizer
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
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
