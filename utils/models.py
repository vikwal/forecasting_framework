import tensorflow as tf
import keras.ops
from keras import layers, optimizers, Model
from typing import Dict, Any, Callable, Tuple
import logging


# --- Helper-Funktionen für Layer-Stacks ---

def _build_rnn_stack(input_tensor: tf.Tensor,
                     layer_type: Callable,
                     units: int,
                     n_layers: int,
                     dropout: float,
                     bidirectional: bool = False,
                     layer_name_prefix: str = 'rnn',
                     final_return_sequences: bool = False) -> tf.Tensor:
    x = input_tensor
    for i in range(n_layers):
        is_last_layer = (i == n_layers - 1)
        return_sequences = (not is_last_layer) or (is_last_layer and final_return_sequences)
        rnn_layer = layer_type(units, dropout=dropout, return_sequences=return_sequences, name=f'{layer_name_prefix}_{i+1}')
        if bidirectional:
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
                            dropout: float,
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
            recurrent_dropout=dropout,
            name=f'{layer_name_prefix}_{i+1}' # Name angepasst
        )
        x = convlstm_layer(x)
        # Optional: Batch Normalization etc.
        # x = layers.BatchNormalization()(x)
    return x

def _grn(x, hidden_dim, context=None, dropout_rate=0.1, name='grn'):
    """
    Gated Residual Network (GRN) - Strikte Implementierung nach Eq. 2-5.
    """
    # --- 1. Context Processing ---
    if context is not None:
        if len(x.shape) == 3 and len(context.shape) == 2:
            time_steps = x.shape[1]
            context = layers.RepeatVector(time_steps)(context)

        context_proj = layers.Dense(hidden_dim, name=f'{name}_ctx_proj')(context)
        x_concat = layers.Concatenate(axis=-1)([x, context_proj])
    else:
        x_concat = x

    # --- 2. Erste Schicht: Dense + ELU (Eq. 4) ---
    # Erzeugt eta_2
    eta_2 = layers.Dense(hidden_dim, activation='elu', name=f'{name}_dense1')(x_concat)

    # --- 3. Zweite Schicht: Linear Dense (Eq. 3) ---
    # Erzeugt eta_1
    eta_1 = layers.Dense(hidden_dim, name=f'{name}_dense2')(eta_2)

    # --- KORREKTUR: Dropout auf eta_1 (Seite 7, Abschnitt 4.1) ---
    eta_1 = layers.Dropout(dropout_rate, name=f'{name}_dropout')(eta_1)

    # --- 4. Gating: GLU (Eq. 2 & 5) ---
    # GLU nimmt eta_1, projiziert es (W4, W5 in Eq. 5) und gatet es.
    # Wir nutzen hier eine Dense Layer mit 2*hidden_dim für W4*gamma und W5*gamma
    glu_input = layers.Dense(hidden_dim * 2, name=f'{name}_glu_dense')(eta_1)
    glu_out = layers.Lambda(lambda z: tf.keras.activations.glu(z, axis=-1), name=f'{name}_glu_act')(glu_input)

    # --- 5. Residual Connection & Norm (Eq. 2) ---
    # Wenn Dimensionen nicht passen (z.B. bei Input Transformation), projiziere Residual
    if x.shape[-1] != hidden_dim:
        x_proj = layers.Dense(hidden_dim, name=f'{name}_res_proj')(x)
        out = layers.Add()([x_proj, glu_out])
    else:
        out = layers.Add()([x, glu_out])

    out = layers.LayerNormalization(name=f'{name}_norm')(out)

    return out


def _variable_selection_network(x, num_features, hidden_dim, context=None, dropout_rate=0.1, name='vsn'):
    """
    Variable Selection Network (VSN) according to Sec. 4.2.

    Shapes:
    - Input x:        (Batch, Time, Num_Features)
    - Input context:  (Batch, Hidden_Dim) [Optional, e.g., c_s]
    - Output:         (Batch, Time, Hidden_Dim)
    - Output Weights: (Batch, Time, Num_Features)
    """

    if num_features == 0:
        return None, None

    # --- Path 1: Generating Selection Weights (Eq. 6) ---
    # The entire input vector is flattened/projected to create a "state"
    # Then fed into a GRN that outputs 'num_features' units (one weight per feature)

    # 1. Flatten/Project state for the Weight-GRN
    # Shape: (Batch, Time, Hidden_Dim)
    flat_input = layers.TimeDistributed(layers.Dense(hidden_dim), name=f'{name}_flat_input')(x)

    # 2. Calculate raw weights via GRN
    # Note: Output dimension here is 'num_features', not 'hidden_dim'!
    # Shape: (Batch, Time, Num_Features)
    raw_weights = _grn(flat_input, hidden_dim=num_features, context=context, dropout_rate=dropout_rate, name=f'{name}_weight_grn')

    # 3. Softmax to ensure weights sum to 1
    # Shape: (Batch, Time, Num_Features)
    feature_weights = layers.TimeDistributed(layers.Softmax(axis=-1), name=f'{name}_softmax')(raw_weights)

    # --- Path 2: Processing Each Feature Independently (Eq. 7) ---
    processed_features = []

    for i in range(num_features):
        # 1. Slice specific feature (keep 3D rank)
        # Shape: (Batch, Time, 1)
        slice_layer = layers.Lambda(lambda z: z[:, :, i:i+1], name=f'{name}_slice_{i}')
        feat_slice = slice_layer(x)

        # 2. Linear Embedding (Scalar -> Vector)
        # Shape: (Batch, Time, Hidden_Dim)
        embedding = layers.TimeDistributed(layers.Dense(hidden_dim), name=f'{name}_emb_{i}')(feat_slice)

        # 3. Feature-specific GRN processing
        # Each feature has its OWN weights in this GRN
        # Shape: (Batch, Time, Hidden_Dim)
        processed = _grn(embedding, hidden_dim=hidden_dim, context=context, dropout_rate=dropout_rate, name=f'{name}_feat_grn_{i}')
        processed_features.append(processed)

    # Stack processed features for efficient multiplication
    # Shape: (Batch, Time, Num_Features, Hidden_Dim)
    processed_stack = layers.Lambda(lambda z: tf.stack(z, axis=2), name=f'{name}_stack')(processed_features)

    # --- Path 3: Weighted Sum (Eq. 8) ---

    # 1. Expand weights for broadcasting
    # Shape: (Batch, Time, Num_Features, 1)
    weights_expanded = layers.Lambda(lambda z: tf.expand_dims(z, axis=-1), name=f'{name}_exp_weights')(feature_weights)

    # 2. Multiply Weights * Processed Features
    # (Batch, Time, Num_Features, 1) * (Batch, Time, Num_Features, Hidden_Dim)
    # Result Shape: (Batch, Time, Num_Features, Hidden_Dim)
    weighted_features = layers.Multiply(name=f'{name}_weight_mult')([processed_stack, weights_expanded])

    # 3. Sum across feature dimension (axis 2)
    # Shape: (Batch, Time, Hidden_Dim)
    vsn_output = layers.Lambda(lambda z: tf.reduce_sum(z, axis=2), name=f'{name}_reduce_sum')(weighted_features)

    return vsn_output, feature_weights

def _static_covariate_encoders(x_static, num_static, hidden_dim, dropout_rate=0.1, name='static_enc'):
    """
    Verarbeitet statische Inputs und erzeugt 4 Kontext-Vektoren.
    Ref: Sec. 4.3

    Input Shapes:
    - x_static: (Batch, Num_Static) -> Raw Inputs

    Output Shapes:
    - c_s, c_e, c_c, c_h: Je (Batch, Hidden_Dim)
    """

    if num_static == 0:
        # Wenn keine statischen Features da sind, geben wir None zurück.
        # Im Build-Prozess müssen wir dann Nullen als Default setzen.
        return None, None, None, None

    # --- 1. Static Variable Selection ---
    # Paper Sec 4.3: "Taking zeta to be the output of the static variable selection network..."

    # Trick: Shape (Batch, Features) -> (Batch, 1, Features) für VSN
    x_expanded = layers.Lambda(lambda z: tf.expand_dims(z, axis=1), name=f'{name}_expand')(x_static)

    # Wende VSN an (Kontext ist hier None, da statische Variablen keinen externen Kontext haben)
    # Output Shape: (Batch, 1, Hidden_Dim)
    static_selection, _ = _variable_selection_network(
        x_expanded, num_static, hidden_dim, context=None, dropout_rate=dropout_rate, name=f'{name}_vsn'
    )

    # Zurück zu 2D: (Batch, Hidden_Dim)
    # Wir nennen diesen Vektor 'zeta' (wie im Paper)
    zeta = layers.Lambda(lambda z: tf.squeeze(z, axis=1), name=f'{name}_squeeze')(static_selection)

    # --- 2. Generate 4 Context Vectors via 4 separate GRNs ---
    # Ref: Sec 4.3, Eq im Text: c_s = GRN(zeta), etc.

    # Context für Variable Selection (wird in ALLE anderen VSNs gespeist)
    c_s = _grn(zeta, hidden_dim, dropout_rate=dropout_rate, name=f'{name}_grn_cs')

    # Context für Static Enrichment (wird vor/nach Attention genutzt)
    c_e = _grn(zeta, hidden_dim, dropout_rate=dropout_rate, name=f'{name}_grn_ce')

    # Contexts für LSTM Initialisierung (Hidden State h und Cell State c)
    c_h = _grn(zeta, hidden_dim, dropout_rate=dropout_rate, name=f'{name}_grn_ch')
    c_c = _grn(zeta, hidden_dim, dropout_rate=dropout_rate, name=f'{name}_grn_cc')

    return c_s, c_e, c_c, c_h

def _locality_enhancement_layer(past_features, future_features, c_c, c_h, hidden_dim, num_lstm_layers=1, dropout_rate=0.1, name='loc_enh'):
    """
    Locality Enhancement via Stacked LSTM Encoder-Decoder (Sec. 4.5.1).

    Neu:
    - num_lstm_layers: Anzahl der gestapelten LSTM Schichten (Default: 1)
    """

    # --- 1. Encoder (Past) ---
    # Wir iterieren durch die gewünschte Anzahl an Layern

    enc_output = past_features
    encoder_final_state = None # Speichert den State des LETZTEN Layers

    for i in range(num_lstm_layers):
        is_first_layer = (i == 0)
        is_last_layer = (i == num_lstm_layers - 1)

        # Initiale States: Nur der erste Layer bekommt den statischen Kontext
        # Alle anderen starten bei 0 (None)
        if is_first_layer:
            current_initial_state = [c_h, c_c] if c_h is not None else None
        else:
            current_initial_state = None

        # Wir brauchen return_state=True NUR beim letzten Layer, um den Decoder zu füttern
        # Wir brauchen return_sequences=True IMMER, weil wir gestapelte LSTMs bauen
        encoder_layer = layers.LSTM(
            hidden_dim,
            return_sequences=True,
            return_state=is_last_layer, # Nur letzter Layer gibt State zurück
            dropout=dropout_rate,
            name=f'{name}_lstm_enc_{i}'
        )

        if is_last_layer:
            enc_output, state_h, state_c = encoder_layer(enc_output, initial_state=current_initial_state)
            encoder_final_state = [state_h, state_c]
        else:
            enc_output = encoder_layer(enc_output, initial_state=current_initial_state)

    # --- 2. Decoder (Future) ---

    dec_output = future_features

    for i in range(num_lstm_layers):
        is_first_layer = (i == 0)

        # Initiale States:
        # Der erste Decoder-Layer übernimmt den State vom letzten Encoder-Layer.
        # Tiefere Decoder-Layer starten bei 0 (Standard Seq2Seq Verhalten).
        if is_first_layer:
            current_initial_state = encoder_final_state
        else:
            current_initial_state = None

        decoder_layer = layers.LSTM(
            hidden_dim,
            return_sequences=True, # Decoder muss immer Sequenz liefern
            dropout=dropout_rate,
            name=f'{name}_lstm_dec_{i}'
        )

        dec_output = decoder_layer(dec_output, initial_state=current_initial_state)

    # --- 3. Recombination & Gating ---
    # Ab hier bleibt alles identisch zum Paper (Concat -> Gating)

    # Combine Past (Encoder Output) + Future (Decoder Output)
    lstm_output = layers.Concatenate(axis=1, name=f'{name}_concat_out')([enc_output, dec_output])

    # Combine Inputs for Residual Connection
    vsn_input_combined = layers.Concatenate(axis=1, name=f'{name}_concat_in')([past_features, future_features])

    # Gated Skip Connection (Eq. 17)
    lstm_gated = layers.Dense(hidden_dim * 2, name=f'{name}_gate_dense')(lstm_output)
    lstm_gated = layers.Lambda(lambda z: tf.keras.activations.glu(z, axis=-1), name=f'{name}_gate_act')(lstm_gated)

    lstm_skip = layers.Add(name=f'{name}_add')([vsn_input_combined, lstm_gated])
    final_output = layers.LayerNormalization(name=f'{name}_norm')(lstm_skip)

    return final_output

def _temporal_self_attention_block(lstm_output, c_e, num_heads, hidden_dim, dropout_rate=0.1, name='temp_att'):
    """
    Führt Static Enrichment, Self-Attention und Feed-Forward aus.
    Gibt den Output des Feed-Forward GRNs zurück (ohne den finalen Skip zum LSTM).
    """

    # --- 1. Static Enrichment Layer (Sec. 4.5.2) ---
    enriched_output = _grn(lstm_output, hidden_dim, context=c_e, dropout_rate=dropout_rate, name=f'{name}_enrich_grn')

    def create_causal_mask(x):
        seq_len = tf.shape(x)[1]
        i = tf.range(seq_len)[:, tf.newaxis]
        j = tf.range(seq_len)[tf.newaxis, :]
        return j <= i

    causal_mask = layers.Lambda(create_causal_mask, name=f'{name}_mask_gen')(enriched_output)

    mha_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=hidden_dim,
        dropout=dropout_rate,
        name=f'{name}_mha'
    )(
        query=enriched_output,
        value=enriched_output,
        key=enriched_output,
        attention_mask=causal_mask
    )

    # Gated Skip Connection (Lokal über Attention)
    mha_gate_in = layers.Dense(hidden_dim * 2, name=f'{name}_mha_gate_dense')(mha_output)
    mha_gated = layers.Lambda(lambda z: tf.keras.activations.glu(z, axis=-1), name=f'{name}_mha_gate_act')(mha_gate_in)
    mha_skip = layers.Add(name=f'{name}_mha_add')([enriched_output, mha_gated])
    att_output_norm = layers.LayerNormalization(name=f'{name}_mha_norm')(mha_skip)

    # --- 3. Position-wise Feed-Forward Layer (Sec. 4.5.4) ---
    # Wir geben direkt das Ergebnis dieses GRNs zurück
    ff_output = _grn(att_output_norm, hidden_dim, context=None, dropout_rate=dropout_rate, name=f'{name}_ff_grn')

    return ff_output

# --- Einzelne Modell-Builder-Funktionen ---

def build_fnn(n_features: int, input_seq_len: int, hp: Dict[str, Any]) -> Model:
    """Baut ein Feedforward Neural Network (FNN)."""
    units = hp.get('units', 32)
    n_layers = hp.get('n_layers', 1) # Anzahl der *hidden* Dense Layer

    input_layer = layers.Input(shape=(input_seq_len, n_features), name='input')
    x = input_layer
    for i in range(n_layers):
         x = layers.Dense(units=units, activation='relu', name=f'dense_{i+1}')(x)
    x = layers.Flatten(name='flatten')(x)
    output_layer = layers.Dense(input_seq_len, name='output')(x)  # Output same as input seq len
    return Model(inputs=input_layer, outputs=output_layer, name='fnn')

def build_conv1d(n_features: int, input_seq_len: int, hp: Dict[str, Any], conv_type: str) -> Model:
    """
    Baut ein einfaches Conv1D-basiertes Modell (CNN oder TCN).
    ERSETZT build_cnn und build_tcn.

    Args:
        n_features: Anzahl der Eingabemerkmale.
        input_seq_len: Länge der Eingabesequenz (Zeitschritte).
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
    input_layer = layers.Input(shape=(input_seq_len, n_features), name='input')

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
    output_layer = layers.Dense(input_seq_len, name='output')(x)  # Output same as input seq len

    # Nutze conv_type als Modellnamen
    model = Model(inputs=input_layer, outputs=output_layer, name=conv_type)
    return model

def build_rnn(n_features: int,
              input_seq_len: int,
              hp: Dict[str, Any],
              layer_type: Callable,
              bidirectional: bool = False,
              name: str = 'rnn') -> Model:
    """Baut ein Long Short-Term Memory (LSTM) Netzwerk."""
    units = hp.get('units', 16)
    n_layers = hp.get('n_rnn_layers', 1)
    dropout = hp.get('dropout', 0)
    input_layer = layers.Input(shape=(input_seq_len, n_features), name='input')
    x = _build_rnn_stack(input_layer, layer_type, units, n_layers, dropout, bidirectional=bidirectional, layer_name_prefix=name)
    output_layer = layers.Dense(input_seq_len, name='output')(x)  # Output same as input seq len
    return Model(inputs=input_layer, outputs=output_layer, name=name)

# --- Hybrid Modell-Builder ---

def build_cnn_rnn(n_features: int, input_seq_len: int, hp: Dict[str, Any],
                  conv_type: str,
                  rnn_layer_type: Callable,
                  rnn_bidirectional: bool,
                  model_name: str) -> Model:
    n_conv_layers = hp.get('n_cnn_layers', 1)
    n_rnn_layers = hp.get('n_rnn_layers', 1)
    filters = hp.get('filters', 16)
    kernel_size = hp.get('kernel_size', 2)
    units = hp.get('units', 16)
    dropout = hp.get('dropout', 0.1)
    increase_filters = hp.get('increase_filters', False)
    conv_activation = hp.get('conv_activation', 'relu')
    use_attention = hp.get('use_attention', False)
    attention_heads = hp.get('attention_heads', 16)

    input_layer = layers.Input(shape=(input_seq_len, n_features), name='input')

    conv_output = _build_conv_stack(
        input_tensor=input_layer,
        filters=filters,
        kernel_size=kernel_size,
        n_layers=n_conv_layers,
        activation=conv_activation,
        conv_type=conv_type,
        increase_filters=increase_filters,
        layer_name_prefix=conv_type
    )
    rnn_output = _build_rnn_stack(
        input_tensor=conv_output,
        layer_type=rnn_layer_type,
        units=units,
        n_layers=n_rnn_layers,
        dropout=dropout,
        bidirectional=rnn_bidirectional,
        layer_name_prefix=model_name.split('-')[-1],
        final_return_sequences=use_attention
    )
    if use_attention:
        attention_dim = units * 2 if rnn_bidirectional else units
        if attention_dim % attention_heads != 0:
            logging.warning(f"Number of attention heads ({attention_heads}) is not a divisor of attention Dim ({attention_dim}). Setting to 1.")
            attention_heads = 1
        key_dim = attention_dim // attention_heads
        attention_layer = layers.MultiHeadAttention(
            num_heads=attention_heads,
            key_dim=key_dim,
            name='multi_head_attention'
        )
        x = attention_layer(query=rnn_output, value=rnn_output, key=rnn_output)
        x = layers.Flatten()(x) # layers.GlobalMaxPooling1D(name='attention_pooling')(x)
    else:
        x = rnn_output
    output_layer = layers.Dense(input_seq_len, name='output')(x)  # Output same as input seq len
    model = Model(inputs=input_layer, outputs=output_layer, name=model_name)
    return model

# Special model builders

def build_convlstm1d(n_features: int,
                     input_seq_len: int,
                     hp: Dict[str, Any]) -> Model:
    """
    Baut ein ConvLSTM1D Netzwerk unter Verwendung von _build_convlstm1d_stack.
    Der ConvLSTM1D-Stack kann als Standard ('convlstm') oder TCN-ähnlich ('tcn')
    konfiguriert werden.
    """
    # Hyperparameter extrahieren mit Defaults
    filters = hp.get('filters', 64)
    kernel_size = hp.get('kernel_size', 3)
    n_layers = hp.get('n_rnn_layers', 1)
    dropout = hp.get('dropout', 0.1)
    activation = hp.get('activation', 'tanh')
    increase_filters = hp.get('increase_filters', False) # Standardmäßig keine Erhöhung

    # Input Layer
    input_layer = layers.Input(shape=(input_seq_len, n_features), name='input')
    # Reshape für ConvLSTM1D
    reshaped_input = layers.Reshape((input_seq_len, 1, n_features),
                                  name='reshape_for_convlstm1d')(input_layer)
    convlstm_output = _build_convlstm1d_stack(
        input_tensor=reshaped_input,
        filters=filters,
        kernel_size=kernel_size,
        n_layers=n_layers,
        dropout=dropout,
        activation=activation,
        increase_filters=increase_filters,
        layer_name_prefix='convlstm1d'
    )
    x = layers.Flatten(name='flatten')(convlstm_output)
    output_layer = layers.Dense(input_seq_len, name='output')(x)  # Output same as input seq len
    model_name = f"convlstm1d"
    model = Model(inputs=input_layer, outputs=output_layer, name=model_name)
    return model

def build_tft(feature_dims: Dict[str, int],
              output_dim: int,
              hyperparameters: Dict[str, Any]):
    """
    Creates a Temporal Fusion Transformer model with Variable Selection,
    Causal Mask, and optional static inputs.

    Implements the full architecture according to Lim et al. (2019),
    including the final skip connection optimization.
    """

    # --- 1. Unpack Dimensions & Hyperparameters ---
    observed_dim = feature_dims['observed_dim']
    known_dim = feature_dims['known_dim']
    static_dim = feature_dims['static_dim']

    n_heads = hyperparameters['n_heads']
    sequence_length = hyperparameters['lookback']
    forecast_horizon = hyperparameters['horizon']
    hidden_dim = hyperparameters['hidden_dim']
    dropout_rate = hyperparameters['dropout']
    # Use .get() for optional parameters with defaults
    n_lstm_layers = hyperparameters.get('n_lstm_layers', 1)
    num_quantiles = hyperparameters.get('num_quantiles', 1)

    # --- 2. Input Layers ---
    observed_input = layers.Input(shape=(sequence_length, observed_dim), name='observed')
    known_input = layers.Input(shape=(sequence_length + forecast_horizon, known_dim), name='known')

    model_inputs = [observed_input, known_input]

    input_static = None
    if static_dim > 0:
        input_static = layers.Input(shape=(static_dim,), name='static')
        model_inputs.append(input_static)

    # --- 3. Static Covariate Encoders (Sec 4.3) ---
    # Generates context vectors for the rest of the network
    c_s, c_e, c_c, c_h = _static_covariate_encoders(
        input_static, static_dim, hidden_dim, dropout_rate, name='static_enc'
    )

    # 1. Known Input aufteilen (ROH-DATEN)
    # Wir schneiden die Rohdaten, bevor sie ins VSN gehen
    raw_known_past = layers.Lambda(lambda x: x[:, :sequence_length, :], name='raw_slice_known_past')(known_input)
    raw_known_future = layers.Lambda(lambda x: x[:, sequence_length:, :], name='raw_slice_known_future')(known_input)

    # 2. Past Input Kombinieren (ROH-DATEN)
    # Observed + Known Past = Alle Vergangenheits-Features
    # Shape: (Batch, Lookback, observed_dim + known_dim)
    raw_past_input = layers.Concatenate(axis=-1, name='raw_concat_past')([observed_input, raw_known_past])

    # Dimension der kombinierten Vergangenheit berechnen für VSN
    total_past_dim = observed_dim + known_dim

    # --- 4. Variable Selection Networks (Sec 4.2) ---
    past_features, _ = _variable_selection_network(
        raw_past_input, total_past_dim, hidden_dim, context=c_s, dropout_rate=dropout_rate, name='vsn_past'
    )

    # b) Future VSN (Verarbeitet nur Known Future)
    # Output: (Batch, Horizon, Hidden)
    future_features, _ = _variable_selection_network(
        raw_known_future, known_dim, hidden_dim, context=c_s, dropout_rate=dropout_rate, name='vsn_future'
    )

    # --- Locality Enhancement ---
    # Nimmt jetzt die sauberen (Batch, T, Hidden) Inputs
    lstm_output = _locality_enhancement_layer(
        past_features,
        future_features,
        c_c, c_h,
        hidden_dim,
        num_lstm_layers=n_lstm_layers,
        dropout_rate=dropout_rate,
        name='loc_enh'
    )

    # --- 5. Locality Enhancement (LSTM Encoder-Decoder) (Sec 4.5.1) ---
    # Processes local patterns. Initialized with static states (c_c, c_h).
    # Returns the full sequence (Past + Future)
    lstm_output = _locality_enhancement_layer(
        past_features,
        future_features,
        c_c, c_h,
        hidden_dim,
        num_lstm_layers=n_lstm_layers,
        dropout_rate=dropout_rate,
        name='loc_enh'
    )

    # --- 6. Temporal Self-Attention Block (Sec 4.5.2 - 4.5.4) ---
    # Contains: Static Enrichment -> Causal Attention -> Feed-Forward
    # Returns ONLY the processed signal (ff_output), without the final skip
    ff_output = _temporal_self_attention_block(
        lstm_output,
        c_e,
        n_heads,
        hidden_dim,
        dropout_rate,
        name='temp_att_block'
    )

    # --- 7. Final Gated Skip Connection (Eq. 22) ---
    # Optimization: Connecting the LSTM output directly to the final block here.
    # Formula: LayerNorm( lstm_output + GLU(ff_output) )

    # a) GLU on the Transformer output
    gate_input = layers.Dense(hidden_dim * 2, name='final_gate_dense')(ff_output)
    gated_ff_output = layers.Lambda(lambda z: tf.keras.activations.glu(z, axis=-1), name='final_gate_act')(gate_input)

    # b) Add: Skip connection from Locality Enhancement (lstm_output)
    final_skip = layers.Add(name='final_skip_add')([lstm_output, gated_ff_output])

    # c) Norm
    final_output = layers.LayerNormalization(name='final_norm')(final_skip)

    # --- 8. Output Head (Sec 4.6) ---
    # Slice to keep only the future horizon for prediction
    future_features_only = layers.Lambda(
        lambda x: x[:, sequence_length:, :],
        name='slice_final_output'
    )(final_output)

    # Final dense layer to project to quantiles (output_dim)
    outputs = layers.TimeDistributed(
        layers.Dense(num_quantiles),
        name='output_quantiles'
    )(future_features_only)

    # --- Model Definition ---
    model = Model(inputs=model_inputs, outputs=outputs, name='TemporalFusionTransformer')

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
    'cnn-lstm': lambda n, o, hp: build_cnn_rnn(n, o, {**hp, 'use_attention': False}, 'cnn', layers.LSTM, False, 'cnn-lstm'),
    'cnn-gru': lambda n, o, hp: build_cnn_rnn(n, o, {**hp, 'use_attention': False}, 'cnn', layers.GRU, False, 'cnn-gru'),
    'cnn-bilstm': lambda n, o, hp: build_cnn_rnn(n, o, {**hp, 'use_attention': False}, 'cnn', layers.LSTM, True, 'cnn-bilstm'),
    'cnn-bigru': lambda n, o, hp: build_cnn_rnn(n, o, {**hp, 'use_attention': False}, 'cnn', layers.GRU, True, 'cnn-bigru'),
    'tcn-lstm': lambda n, o, hp: build_cnn_rnn(n, o, {**hp, 'use_attention': False}, 'tcn', layers.LSTM, False, 'tcn-lstm'),
    'tcn-gru': lambda n, o, hp: build_cnn_rnn(n, o, {**hp, 'use_attention': False}, 'tcn', layers.GRU, False, 'tcn-gru'),
    'tcn-bilstm': lambda n, o, hp: build_cnn_rnn(n, o, {**hp, 'use_attention': False}, 'tcn', layers.LSTM, True, 'tcn-bilstm'),
    'tcn-bigru': lambda n, o, hp: build_cnn_rnn(n, o, {**hp, 'use_attention': False}, 'tcn', layers.GRU, True, 'tcn-bigru'),
    # hybrid models with attention
    'cnn-lstm-attn': lambda n, o, hp: build_cnn_rnn(n, o, {**hp, 'use_attention': True}, 'cnn', layers.LSTM, False, 'cnn-lstm-attn'),
    'cnn-gru-attn': lambda n, o, hp: build_cnn_rnn(n, o, {**hp, 'use_attention': True}, 'cnn', layers.GRU, False, 'cnn-gru-attn'),
    'cnn-bilstm-attn': lambda n, o, hp: build_cnn_rnn(n, o, {**hp, 'use_attention': True}, 'cnn', layers.LSTM, True, 'cnn-bilstm-attn'),
    'cnn-bigru-attn': lambda n, o, hp: build_cnn_rnn(n, o, {**hp, 'use_attention': True}, 'cnn', layers.GRU, True, 'cnn-bigru-attn'),
    'tcn-lstm-attn': lambda n, o, hp: build_cnn_rnn(n, o, {**hp, 'use_attention': True}, 'tcn', layers.LSTM, True, 'tcn-lstm-attn'),
    'tcn-gru-attn': lambda n, o, hp: build_cnn_rnn(n, o, {**hp, 'use_attention': True}, 'tcn', layers.GRU, False, 'tcn-gru-attn'),
    'tcn-bilstm-attn': lambda n, o, hp: build_cnn_rnn(n, o, {**hp, 'use_attention': True}, 'tcn', layers.LSTM, True, 'tcn-bilstm-attn'),
    'tcn-bigru-attn': lambda n, o, hp: build_cnn_rnn(n, o, {**hp, 'use_attention': True}, 'tcn', layers.GRU, True, 'tcn-bigru-attn'),
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
        elif m == 'r^2': metrics.append(tf.keras.metrics.R2Score(name='r2'))
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

    # CORRECT: Use feature_dim as number of features, output_dim as sequence length
    model = builder(feature_dim, output_dim, hyperparameters)

    # optimizer
    optimizer_name = config.get('model', {}).get('optimizer', 'adam') # Default zu Adam
    learning_rate = hyperparameters.get('lr', 0.001) # Default Learning Rate
    clipnorm = hyperparameters.get('clipnorm', 1.0) # Default Clipnorm
    optimizer_class = OPTIMIZERS.get(optimizer_name)
    if not optimizer_class:
        raise ValueError(f"Unbekannter Optimizer: '{optimizer_name}'. Verfügbar: {list(OPTIMIZERS.keys())}")

    optimizer = optimizer_class(learning_rate=learning_rate, clipnorm=clipnorm)

    # Kompiliere das Modell
    loss = config.get('model', {}).get('loss', 'mse') # Default zu MSE
    metrics = config.get('model', {}).get('metrics', ['mae']) # Default zu MAE
    metrics = get_metrics(config=config)
    # Sicherstellen, dass Metriken eine Liste ist, falls nur ein String übergeben wird
    if isinstance(metrics, str):
        metrics = [metrics]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
