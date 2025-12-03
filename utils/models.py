"""
implementation of Temporal Fusion Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any


def get_model(config: Dict[str, Any], hyperparameters: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create PyTorch models based on config.
    Matches utils/models.py get_model() signature.
    """
    model_name = config['model']['name']

    if model_name == 'tft':
        feature_dims = config['model']['feature_dim']

        model = TFT(
            observed_dim=feature_dims['observed_dim'],
            known_dim=feature_dims['known_dim'],
            static_dim=feature_dims['static_dim'],
            hidden_dim=hyperparameters.get('hidden_size', hyperparameters.get('hidden_dim', 32)),
            num_heads=hyperparameters.get('attention_head_size', hyperparameters.get('n_heads', 4)),
            lookback=config['model']['lookback'],
            horizon=config['model']['horizon'],
            dropout=hyperparameters.get('dropout', 0.1),
            num_lstm_layers=hyperparameters.get('num_lstm_layers', hyperparameters.get('n_lstm_layers', 1)),
            static_embedding_dim=hyperparameters.get('static_embedding_dim', None),
        )
        return model

    elif model_name == 'cnn-lstm':
        feature_dim = config['model']['feature_dim']
        output_dim = config['model']['output_dim']

        model = CNNRNN(
            n_features=feature_dim,
            output_dim=output_dim,
            conv_type='cnn',
            rnn_type='lstm',
            bidirectional=False,
            hyperparameters=hyperparameters
        )
        return model

    elif model_name == 'tcn-gru':
        feature_dim = config['model']['feature_dim']
        output_dim = config['model']['output_dim']

        model = CNNRNN(
            n_features=feature_dim,
            output_dim=output_dim,
            conv_type='tcn',
            rnn_type='gru',
            bidirectional=False,
            hyperparameters=hyperparameters
        )
        return model

    else:
        raise NotImplementedError(f"Model {model_name} not implemented in PyTorch yet")


# --- Helper Functions for Building Layer Stacks ---

def _build_conv_stack(
    input_channels: int,
    filters: int,
    kernel_size: int,
    n_layers: int,
    activation: str = 'relu',
    conv_type: str = 'cnn',
    increase_filters: bool = False
) -> nn.Sequential:
    """
    Builds a stack of Conv1D layers, supports 'cnn' or 'tcn' behavior.
    Matches the TensorFlow _build_conv_stack function.

    Args:
        input_channels: Number of input channels (features)
        filters: Initial number of filters for the first layer
        kernel_size: Size of the convolutional kernel
        n_layers: Number of Conv1D layers
        activation: Activation function ('relu', 'tanh', etc.)
        conv_type: 'cnn' for standard Conv1D (padding='same') or
                   'tcn' for TCN-like (causal padding, increasing dilation)
        increase_filters: If True, increase filter count as filters * (2**i) for layer i > 0

    Returns:
        Sequential module containing the conv stack
    """
    layers = []

    # Get activation function
    if activation == 'relu':
        act_fn = nn.ReLU()
    elif activation == 'tanh':
        act_fn = nn.Tanh()
    elif activation == 'elu':
        act_fn = nn.ELU()
    else:
        act_fn = nn.ReLU()

    for i in range(n_layers):
        # Determine padding and dilation based on conv_type
        if conv_type == 'tcn':
            dilation = 2 ** i
            # Causal padding: pad only on the left to prevent information leakage
            padding = (kernel_size - 1) * dilation
        else:  # conv_type == 'cnn'
            dilation = 1
            padding = kernel_size // 2  # 'same' padding

        # Determine number of filters for current layer
        layer_filters = filters * (2 ** i) if (increase_filters and i > 0) else filters

        # Determine input channels
        in_channels = input_channels if i == 0 else (filters * (2 ** (i-1)) if increase_filters else filters)

        # Add Conv1D layer
        if conv_type == 'tcn':
            # For TCN, we need causal padding on the left only
            layers.append(nn.ConstantPad1d((padding, 0), 0))  # Pad left only
            layers.append(nn.Conv1d(in_channels, layer_filters, kernel_size, dilation=dilation))
        else:
            layers.append(nn.Conv1d(in_channels, layer_filters, kernel_size, padding=padding, dilation=dilation))

        layers.append(act_fn)

    return nn.Sequential(*layers)


def _build_rnn_stack(
    input_size: int,
    hidden_size: int,
    n_layers: int,
    dropout: float,
    rnn_type: str = 'lstm',
    bidirectional: bool = False,
    final_return_sequences: bool = False
) -> nn.Module:
    """
    Builds a stack of RNN layers (LSTM or GRU).
    Matches the TensorFlow _build_rnn_stack function.

    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        n_layers: Number of RNN layers
        dropout: Dropout rate (applied between layers if n_layers > 1)
        rnn_type: Type of RNN ('lstm' or 'gru')
        bidirectional: Whether to use bidirectional RNN
        final_return_sequences: If True, return all time steps; if False, return only last

    Returns:
        RNN module
    """
    if rnn_type.lower() == 'lstm':
        rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )
    elif rnn_type.lower() == 'gru':
        rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )
    else:
        raise ValueError(f"Unknown RNN type: {rnn_type}")

    return rnn


# --- Model Classes ---

class GLU(nn.Module):
    """Gated Linear Unit

    Args:
        input_size: Input dimension
        output_size: Output dimension (defaults to input_size if not specified)
        dropout: Dropout rate to apply before gating (default: 0.0)
    """
    def __init__(self, input_size: int, output_size: int = None, dropout: float = 0.0):
        super().__init__()
        if output_size is None:
            output_size = input_size
        self.fc = nn.Linear(input_size, output_size * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        # Apply dropout before gating (matches TensorFlow apply_gating_layer)
        if self.dropout is not None:
            x = self.dropout(x)
        return F.glu(self.fc(x), dim=-1)


class GRN(nn.Module):
    """Gated Residual Network - matches TensorFlow implementation exactly"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.1, context_size: Optional[int] = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size

        # Primary processing layers
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Context projection (added to fc1 output, not concatenated)
        if context_size is not None:
            self.context_proj = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.context_proj = None

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # GLU gating layer
        self.glu = GLU(hidden_size, output_size)

        if input_size != output_size:
            self.skip_fc = nn.Linear(input_size, output_size)
        else:
            self.skip_fc = None

        self.ln = nn.LayerNorm(output_size)

    def forward(self, x, context=None):
        # --- 1. First Dense layer (Eq. 4 part 1) ---
        # In TensorFlow: hidden = linear_layer(hidden_layer_size)(x)
        eta_2 = self.fc1(x)

        # --- 2. Add context if present (Eq. 4 part 2) ---
        # In TensorFlow: hidden = hidden + linear_layer(hidden_layer_size, use_bias=False)(context)
        if context is not None and self.context_proj is not None:
            # Context might not have time dimension - expand if needed
            if len(context.shape) == 2 and len(x.shape) == 3:
                # context: (batch, context_dim), x: (batch, time, input_dim)
                # Expand context to (batch, 1, context_dim) and repeat for time steps
                time_steps = x.size(1)
                context = context.unsqueeze(1).repeat(1, time_steps, 1)

            # Project and ADD context (not concatenate!)
            eta_2 = eta_2 + self.context_proj(context)

        # --- 3. ELU activation (Eq. 4 part 3) ---
        # In TensorFlow: hidden = Activation('elu')(hidden)
        eta_2 = F.elu(eta_2)

        # --- 4. Second Dense (Eq. 3) ---
        eta_1 = self.fc2(eta_2)

        # --- 5. Dropout ---
        eta_1 = self.dropout(eta_1)

        # --- 6. Gating with GLU (Eq. 2 & 5) ---
        gated = self.glu(eta_1)

        # --- 7. Skip connection ---
        if self.skip_fc is not None:
            x = self.skip_fc(x)

        # --- 8. Residual + LayerNorm ---
        return self.ln(x + gated)


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable Multi-Head Attention from TFT Paper.

    Key difference from standard multi-head attention:
    All heads share the SAME value projection, which makes the attention
    patterns more interpretable and easier to visualize.

    Args:
        n_head: Number of attention heads
        d_model: Model dimension (must be divisible by n_head)
        dropout: Dropout rate
    """
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head  # Dimension per head
        self.d_v = d_model // n_head

        # Separate Q and K projections for each head
        self.w_qs = nn.ModuleList([
            nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)
        ])
        self.w_ks = nn.ModuleList([
            nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)
        ])

        # SHARED value projection for ALL heads (interpretability!)
        self.w_v = nn.Linear(d_model, self.d_v, bias=False)

        # Output projection (from averaged d_v to d_model)
        self.w_o = nn.Linear(self.d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        """Forward pass for interpretable multi-head attention.

        Args:
            q: Query tensor (batch, time, d_model)
            k: Key tensor (batch, time, d_model)
            v: Value tensor (batch, time, d_model)
            attn_mask: Optional attention mask (batch, time, time) or (time, time)
                  True values are MASKED (set to -inf before softmax)

        Returns:
            output: Attention output (batch, time, d_model)
            attn_weights: Attention weights (n_head, batch, time_q, time_k)
        """
        batch_size = q.size(0)

        # SHARED value projection for all heads
        v_proj = self.w_v(v)  # (batch, time, d_v)

        # Process each head
        heads = []
        attn_weights_list = []

        for i in range(self.n_head):
            # Separate Q and K for each head
            q_i = self.w_qs[i](q)  # (batch, time_q, d_k)
            k_i = self.w_ks[i](k)  # (batch, time_k, d_k)

            # Scaled dot-product attention
            # scores: (batch, time_q, time_k)
            scores = torch.matmul(q_i, k_i.transpose(-2, -1)) / (self.d_k ** 0.5)

            # Apply mask if provided
            if attn_mask is not None:
                # Expand mask for batch dimension if needed
                if attn_mask.dim() == 2:  # (time_q, time_k)
                    attn_mask_expanded = attn_mask.unsqueeze(0)  # (1, time_q, time_k)
                else:
                    attn_mask_expanded = attn_mask
                # Set masked positions to large negative value
                scores = scores.masked_fill(attn_mask_expanded, -1e9)

            # Softmax to get attention weights
            attn = torch.softmax(scores, dim=-1)  # (batch, time_q, time_k)

            # Apply attention to SHARED value projection
            head_output = torch.matmul(attn, v_proj)  # (batch, time_q, d_v)

            # Apply dropout to HEAD OUTPUT (matches TensorFlow: head_dropout = Dropout(self.dropout)(head))
            head_output = self.dropout(head_output)

            heads.append(head_output)
            attn_weights_list.append(attn)

        # Average across heads (instead of concatenation)
        # This is specific to the TFT interpretable attention
        multi_head = torch.stack(heads, dim=0)  # (n_head, batch, time_q, d_v)
        output = torch.mean(multi_head, dim=0)  # (batch, time_q, d_v)

        # Output projection
        output = self.w_o(output)  # (batch, time_q, d_model)
        output = self.dropout(output)

        # Stack attention weights for interpretability
        attn_weights = torch.stack(attn_weights_list, dim=0)  # (n_head, batch, time_q, time_k)

        return output, attn_weights


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network

    Implements variable selection as described in the TFT paper.
    Expects PRE-EMBEDDED features (already transformed to embedding dimension).

    Args:
        input_size: Embedding dimension of each input variable (d_model in paper)
        num_inputs: Number of input variables to select from
        hidden_size: Hidden dimension for GRN processing
        dropout: Dropout rate
        context_size: Optional context dimension (e.g., from static features)

    Input:
        x: Pre-embedded features, shape (batch, time, num_inputs, input_size)
           where input_size is the embedding dimension (e.g., hidden_dim)

    Output:
        output: Selected and combined features (batch, time, hidden_size)
        weights: Selection weights (batch, time, num_inputs)
    """
    def __init__(self, input_size: int, num_inputs: int, hidden_size: int,
                 dropout: float = 0.1, context_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_inputs = num_inputs

        if num_inputs > 1:
            # Flattening layer - uses context for weight computation
            self.flattened_grn = GRN(
                input_size=num_inputs * input_size,
                hidden_size=hidden_size,
                output_size=num_inputs,
                dropout=dropout,
                context_size=context_size  # ← Context only here!
            )

            # Per-variable processing - NO context!
            self.single_variable_grns = nn.ModuleList([
                GRN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    dropout=dropout,
                    context_size=None  # ← NO context for individual variables!
                ) for _ in range(num_inputs)
            ])
        else:
            # Single feature - no selection needed, no context
            self.single_variable_grns = nn.ModuleList([
                GRN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    dropout=dropout,
                    context_size=None  # ← NO context
                )
            ])

    def forward(self, x, context=None):
        # x shape: (batch, time, num_inputs * input_size) or (batch, time, num_inputs, input_size)
        batch_size = x.size(0)
        time_steps = x.size(1)

        if self.num_inputs > 1:
            # Flatten if needed
            if len(x.shape) == 4:  # (batch, time, num_inputs, input_size)
                x_flat = x.reshape(batch_size, time_steps, -1)
            else:
                x_flat = x

            # Calculate selection weights WITH context
            weights = self.flattened_grn(x_flat, context)
            weights = F.softmax(weights, dim=-1)

            # Process each variable WITHOUT context
            var_outputs = []
            for i, grn in enumerate(self.single_variable_grns):
                var_outputs.append(
                    grn(x[..., i, :] if len(x.shape) == 4 else x[..., i:i+1])  # ← NO context!
                )

            # Stack and weight
            var_outputs = torch.stack(var_outputs, dim=-2)  # (batch, time, num_inputs, hidden)
            weights = weights.unsqueeze(-1)  # (batch, time, num_inputs, 1)

            output = torch.sum(var_outputs * weights, dim=-2)  # (batch, time, hidden)

        else:
            # Single input - no context
            output = self.single_variable_grns[0](x.squeeze(-2) if len(x.shape) == 4 else x)
            weights = torch.ones(batch_size, time_steps, 1, device=x.device)

        return output, weights


class TFT(nn.Module):
    """Temporal Fusion Transformer in PyTorch"""

    def __init__(
        self,
        observed_dim: int,
        known_dim: int,
        static_dim: int,
        hidden_dim: int,
        num_heads: int,
        lookback: int,
        horizon: int,
        dropout: float = 0.1,
        num_lstm_layers: int = 1,
        static_embedding_dim: Optional[int] = None,
    ):
        super().__init__()

        self.observed_dim = observed_dim
        self.known_dim = known_dim
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim
        self.lookback = lookback
        self.horizon = horizon
        self.num_lstm_layers = num_lstm_layers

        # Use separate embedding dimension for static features if specified
        # Otherwise use same as hidden_dim (backward compatible)
        self.static_embedding_dim = static_embedding_dim if static_embedding_dim is not None else hidden_dim

        # === Feature Embedding Layers (Paper: raw → ξ) ===
        # Transform raw features to d_model dimension BEFORE variable selection

        # Static feature embeddings
        if static_dim > 0:
            self.static_embed = nn.ModuleList([
                nn.Linear(1, self.static_embedding_dim) for _ in range(static_dim)
            ])

        # Observed feature embeddings (time-varying, unknown future)
        self.observed_embed = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(observed_dim)
        ])

        # Known feature embeddings (time-varying, known future)
        self.known_embed = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(known_dim)
        ])

        # === Static Processing ===
        # Static covariate encoders
        if static_dim > 0:
            self.static_variable_selection = VariableSelectionNetwork(
                input_size=self.static_embedding_dim,  # ← Now expects embeddings!
                num_inputs=static_dim,
                hidden_size=self.static_embedding_dim,
                dropout=dropout
            )

            # Context vectors - project from static embedding to hidden dim
            self.static_context_grn = GRN(self.static_embedding_dim, hidden_dim, hidden_dim, dropout)
            self.static_enrichment_grn = GRN(self.static_embedding_dim, hidden_dim, hidden_dim, dropout)
            self.static_state_h_grn = GRN(self.static_embedding_dim, hidden_dim, hidden_dim, dropout)
            self.static_state_c_grn = GRN(self.static_embedding_dim, hidden_dim, hidden_dim, dropout)

        # === Temporal Variable Selection ===
        # Variable selection networks (now operate on embedded features)
        self.vsn_past = VariableSelectionNetwork(
            input_size=hidden_dim,  # ← Now expects embeddings!
            num_inputs=observed_dim + known_dim,
            hidden_size=hidden_dim,
            dropout=dropout,
            context_size=hidden_dim if static_dim > 0 else None
        )

        self.vsn_future = VariableSelectionNetwork(
            input_size=hidden_dim,  # ← Now expects embeddings!
            num_inputs=known_dim,
            hidden_size=hidden_dim,
            dropout=dropout,
            context_size=hidden_dim if static_dim > 0 else None
        )

        # LSTM encoder-decoder (same number of layers for both)
        self.num_encoder_layers = num_lstm_layers
        self.num_decoder_layers = num_lstm_layers
        self.lstm_encoder = nn.LSTM(
            hidden_dim, hidden_dim, num_lstm_layers,
            batch_first=True, dropout=dropout if num_lstm_layers > 1 else 0
        )
        self.lstm_decoder = nn.LSTM(
            hidden_dim, hidden_dim, num_lstm_layers,
            batch_first=True, dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Gated skip connection for LSTM (with dropout like TensorFlow)
        self.lstm_gate = GLU(hidden_dim, dropout=dropout)
        self.lstm_ln = nn.LayerNorm(hidden_dim)

        # Static enrichment
        self.enrichment_grn = GRN(
            hidden_dim, hidden_dim, hidden_dim, dropout,
            context_size=hidden_dim if static_dim > 0 else None
        )

        # Self-attention (interpretable multi-head attention from paper)
        self.multihead_attn = InterpretableMultiHeadAttention(
            n_head=num_heads,
            d_model=hidden_dim,
            dropout=dropout
        )
        self.attn_gate = GLU(hidden_dim, dropout=dropout)  # ← Dropout added!
        self.attn_ln = nn.LayerNorm(hidden_dim)

        # Position-wise feed-forward
        self.positionwise_grn = GRN(hidden_dim, hidden_dim, hidden_dim, dropout)

        # Final gated skip (NO dropout like TensorFlow)
        self.output_gate = GLU(hidden_dim)  # ← No dropout parameter!
        self.output_ln = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, observed, known, static=None, return_attention_weights=False):
        """Forward pass of TFT.

        Args:
            observed: Observed features (batch, lookback, observed_dim)
            known: Known features (batch, lookback+horizon, known_dim)
            static: Static features (batch, static_dim), optional
            return_attention_weights: If True, return attention weights for interpretability

        Returns:
            predictions: (batch, horizon)
            attention_weights: Dict of attention weights (if return_attention_weights=True)
        """
        batch_size = observed.size(0)

        # ===================================================================
        # STEP 1: STATIC FEATURE EMBEDDING (raw → ξ_static)
        # ===================================================================
        if static is not None and self.static_dim > 0:
            # static shape: (batch, static_dim)
            # Embed each static feature: scalar → d_model
            static_embedded = []
            for i in range(self.static_dim):
                feat = static[:, i:i+1]  # (batch, 1)
                embedded = self.static_embed[i](feat)  # (batch, static_emb_dim)
                static_embedded.append(embedded)

            # Stack: (batch, static_dim, static_emb_dim)
            static_embedded = torch.stack(static_embedded, dim=1)

            # Add time dimension for VSN: (batch, 1, static_dim, static_emb_dim)
            static_embedded = static_embedded.unsqueeze(1)

            # Variable selection on EMBEDDED features - KEEP weights!
            static_encoded, static_weights = self.static_variable_selection(static_embedded, None)
            static_encoded = static_encoded.squeeze(1)  # (batch, static_emb_dim)

            # Generate context vectors
            c_selection = self.static_context_grn(static_encoded)
            c_enrichment = self.static_enrichment_grn(static_encoded)
            c_state_h = self.static_state_h_grn(static_encoded)
            c_state_c = self.static_state_c_grn(static_encoded)
        else:
            c_selection = None
            c_enrichment = None
            c_state_h = None
            c_state_c = None
            static_weights = None

        # ===================================================================
        # STEP 2: TEMPORAL FEATURE EMBEDDING (raw → ξ_temporal)
        # ===================================================================
        # Split known features
        known_past = known[:, :self.lookback, :]  # (batch, lookback, known_dim)
        known_future = known[:, self.lookback:, :]  # (batch, horizon, known_dim)

        # Embed observed features (time-varying, unknown future)
        # observed: (batch, lookback, observed_dim)
        observed_embedded = []
        for i in range(self.observed_dim):
            feat = observed[..., i:i+1]  # (batch, lookback, 1)
            embedded = self.observed_embed[i](feat)  # (batch, lookback, hidden_dim)
            observed_embedded.append(embedded)
        observed_embedded = torch.stack(observed_embedded, dim=-2) # (batch, lookback, observed_dim, hidden_dim)

        # Embed known_past features (time-varying, known future)
        # known_past: (batch, lookback, known_dim)
        known_past_embedded = []
        for i in range(self.known_dim):
            feat = known_past[..., i:i+1]  # (batch, lookback, 1)
            embedded = self.known_embed[i](feat)  # (batch, lookback, hidden_dim)
            known_past_embedded.append(embedded)
        known_past_embedded = torch.stack(known_past_embedded, dim=-2) # (batch, lookback, known_dim, hidden_dim)

        # Embed known_future features
        # known_future: (batch, horizon, known_dim)
        known_future_embedded = []
        for i in range(self.known_dim):
            feat = known_future[..., i:i+1]  # (batch, horizon, 1)
            embedded = self.known_embed[i](feat)  # (batch, horizon, hidden_dim)
            known_future_embedded.append(embedded)
        known_future_embedded = torch.stack(known_future_embedded, dim=-2) # (batch, horizon, known_dim, hidden_dim)

        # Concatenate observed and known_past embedded features for past VSN
        past_input_embedded = torch.cat([observed_embedded, known_past_embedded], dim=-2) # (batch, lookback, observed+known, hidden_dim)

        # ===================================================================
        # STEP 3: TEMPORAL VARIABLE SELECTION (ξ_temporal → V_temporal)
        # ===================================================================
        # KEEP weights for interpretability!
        past_encoded, past_weights = self.vsn_past(past_input_embedded, c_selection) # (batch, lookback, hidden_dim)
        future_encoded, future_weights = self.vsn_future(known_future_embedded, c_selection) # (batch, horizon, hidden_dim)

        # ===================================================================
        # STEP 4: LSTM ENCODER-DECODER
        # ===================================================================
        initial_encoder_state = None
        if static is not None and self.static_dim > 0:
            state_h = c_state_h.unsqueeze(0).repeat(self.num_encoder_layers, 1, 1)
            state_c = c_state_c.unsqueeze(0).repeat(self.num_encoder_layers, 1, 1)
            initial_encoder_state = (state_h, state_c)

        # Encode past and decode future
        encoder_output, encoder_state = self.lstm_encoder(past_encoded, initial_encoder_state)
        decoder_output, _ = self.lstm_decoder(future_encoded, encoder_state)

        lstm_output = torch.cat([encoder_output, decoder_output], dim=1)

        # Gated skip
        vsn_combined = torch.cat([past_encoded, future_encoded], dim=1)
        lstm_gated = self.lstm_gate(lstm_output)
        lstm_output = self.lstm_ln(vsn_combined + lstm_gated)

        # Static enrichment
        enriched = self.enrichment_grn(lstm_output, c_enrichment)

        # Self-attention with causal mask
        seq_len = lstm_output.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=lstm_output.device), diagonal=1).bool()

        # KEEP attention weights for interpretability!
        attn_output, attn_weights = self.multihead_attn(
            enriched, enriched, enriched,
            attn_mask=causal_mask
        )

        attn_gated = self.attn_gate(attn_output)
        attn_output = self.attn_ln(enriched + attn_gated)

        # Position-wise feed-forward
        ff_output = self.positionwise_grn(attn_output)

        # Final skip connection
        final_gated = self.output_gate(ff_output)
        final_output = self.output_ln(lstm_output + final_gated)

        # Extract future part and project
        future_output = final_output[:, self.lookback:, :]
        predictions = self.output_layer(future_output).squeeze(-1)

        # Return with attention weights if requested
        if return_attention_weights:
            attention_dict = {
                'decoder_self_attn': attn_weights,                               # (n_head, batch, time, time)
                'static_weights': static_weights.squeeze(-1) if static_weights is not None else None,  # (batch, 1, num_static)
                'past_weights': past_weights.squeeze(-1),                        # (batch, lookback, num_past_inputs)
                'future_weights': future_weights.squeeze(-1)                     # (batch, horizon, num_future_inputs)
            }
            return predictions, attention_dict
        else:
            return predictions


class CNNRNN(nn.Module):
    """
    Hybrid CNN-RNN model in PyTorch.
    Supports CNN-LSTM, CNN-GRU, TCN-LSTM, TCN-GRU variants.
    Matches the TensorFlow build_cnn_rnn implementation.
    """

    def __init__(
        self,
        n_features: int,
        output_dim: int,
        conv_type: str,
        rnn_type: str,
        bidirectional: bool,
        hyperparameters: Dict[str, Any]
    ):
        super().__init__()

        self.n_features = n_features
        self.output_dim = output_dim
        self.conv_type = conv_type
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        # Extract hyperparameters
        self.n_conv_layers = hyperparameters.get('n_cnn_layers', 1)
        self.n_rnn_layers = hyperparameters.get('n_rnn_layers', 1)
        self.filters = hyperparameters.get('filters', 16)
        self.kernel_size = hyperparameters.get('kernel_size', 2)
        self.units = hyperparameters.get('units', 16)
        self.dropout = hyperparameters.get('dropout', 0.1)
        self.increase_filters = hyperparameters.get('increase_filters', False)
        self.conv_activation = hyperparameters.get('conv_activation', 'relu')
        self.use_attention = hyperparameters.get('use_attention', False)
        self.attention_heads = hyperparameters.get('attention_heads', 16)

        # Conv stack
        self.conv_stack = _build_conv_stack(
            input_channels=n_features,
            filters=self.filters,
            kernel_size=self.kernel_size,
            n_layers=self.n_conv_layers,
            activation=self.conv_activation,
            conv_type=conv_type,
            increase_filters=self.increase_filters
        )

        # Calculate conv output channels
        if self.increase_filters and self.n_conv_layers > 1:
            conv_output_channels = self.filters * (2 ** (self.n_conv_layers - 1))
        else:
            conv_output_channels = self.filters

        # RNN stack
        self.rnn = _build_rnn_stack(
            input_size=conv_output_channels,
            hidden_size=self.units,
            n_layers=self.n_rnn_layers,
            dropout=self.dropout,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            final_return_sequences=self.use_attention
        )

        # Attention layer (optional)
        if self.use_attention:
            attention_dim = self.units * 2 if bidirectional else self.units
            if attention_dim % self.attention_heads != 0:
                self.attention_heads = 1
            self.attention = nn.MultiheadAttention(
                embed_dim=attention_dim,
                num_heads=self.attention_heads,
                dropout=self.dropout,
                batch_first=True
            )

        # Output layer
        if self.use_attention:
            # After attention, we flatten the sequence
            # This is a placeholder - actual input size depends on sequence length
            # We'll handle this dynamically in forward pass
            self.output_fc = None  # Will be created in forward pass
        else:
            rnn_output_size = self.units * 2 if bidirectional else self.units
            self.output_fc = nn.Linear(rnn_output_size, output_dim)

    def forward(self, x):
        # x shape: (batch, time, features)
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Conv expects (batch, channels, time)
        x = x.transpose(1, 2)  # (batch, features, time)

        # Conv stack
        x = self.conv_stack(x)

        # Back to (batch, time, channels) for RNN
        x = x.transpose(1, 2)

        # RNN stack
        if self.use_attention:
            # Return all sequences
            rnn_output, _ = self.rnn(x)

            # Self-attention
            attn_output, _ = self.attention(rnn_output, rnn_output, rnn_output)

            # Flatten for output
            attn_output_flat = attn_output.reshape(batch_size, -1)

            # Create output layer dynamically if needed
            if self.output_fc is None:
                input_size = attn_output_flat.size(1)
                self.output_fc = nn.Linear(input_size, self.output_dim).to(x.device)

            output = self.output_fc(attn_output_flat)
        else:
            # Return only last time step
            rnn_output, _ = self.rnn(x)
            # Take last time step
            last_output = rnn_output[:, -1, :]
            output = self.output_fc(last_output)

        return output
