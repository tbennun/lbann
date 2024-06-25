"""
GPT-NeoX transformer variant.
"""
import lbann
from lbann.modules.transformer import encoding, normalization, attention
import math
import numpy as np
from typing import Optional, Type
import warnings


class GPTNeoX(lbann.modules.Module):
    """
    Implements a GPT-NeoX transformer variant.

    For more information, see https://github.com/EleutherAI/gpt-neox and the
    paper: S. Black et al., "GPT-NeoX-20B: An Open-Source Autoregressive
    Language Model," Proceedings of the ACL Workshop on Challenges &
    Perspectives in Creating Large Language Models, 2022.
    """

    def __init__(
        self,
        # Model parameters (set to GPT-NeoX-20B)
        vocab_size: int = 50432,
        hidden_size: int = 6144,
        num_layers: int = 44,
        num_heads: int = 64,
        intermediate_size: int = 24576,
        activation: Type[lbann.Layer] = lbann.Gelu,
        sequence_length: int = 2048,
        # Attention parameters
        parallel_mlp: bool = True,
        attention_bias: bool = True,
        # Dropout
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        # Positional encoding
        positional_encoding: encoding.SequenceEncoding = None,
        # Initialization
        layer_norm_eps: float = 1e-5,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.input_dropout = input_dropout
        self.sequence_length = sequence_length
        self.separate_heads = False
        self._subsequent_mask_cache = {}
        self.name = 'neox'

        # Initialization (from paper)
        small_init = math.sqrt(2 / (5 * hidden_size))
        wang_init = 2 / num_layers / math.sqrt(hidden_size)

        if not isinstance(positional_encoding,
                          encoding.RotaryPositionalEmbedding):
            warnings.warn(
                'GPT-NeoX was designed to train with Rotary Positional Embedding'
            )

        self.layers = [
            GPTNeoXLayer(hidden_size, num_heads, activation, intermediate_size,
                         layer_norm_eps, dropout, attn_dropout, parallel_mlp,
                         attention_bias, small_init, wang_init,
                         sequence_length, positional_encoding, f'layer_{i}')
            for i in range(num_layers)
        ]

        # Embedding weights
        self.embedding_weights = lbann.Weights(
            name=f'{self.name}_embeddings',
            initializer=lbann.NormalInitializer(standard_deviation=small_init),
        )

    def _subsequent_mask(self, size):
        """Attention mask to prevent attending to subsequent positions.

        The (i,j) entry is -1e9 if i<j and is 0 otherwise. Masks are
        memoized.

        """

        # Construct mask if not in cache
        if size not in self._subsequent_mask_cache:
            vals = np.triu(np.full((size, size), -1e9), k=1)

            if not self.separate_heads:
                # Precompute mask for all heads because Add is entry-wise
                # (potential memory usage issue)
                vals = np.tile(vals, (self.num_heads, 1, 1))

            weights = lbann.Weights(
                initializer=lbann.ValueInitializer(values=vals.flat),
                optimizer=lbann.NoOptimizer(),
                name=f'{self.name}_mask{size}_weights',
            )
            self._subsequent_mask_cache[size] = lbann.WeightsLayer(
                dims=vals.shape,
                weights=weights,
                name=f'{self.name}_mask{size}',
            )

        # Return cached mask
        return self._subsequent_mask_cache[size]

    def forward(self, x: lbann.Layer, attn_mask: Optional[lbann.Layer] = None):
        # NOTE: Positional encoding applied on input is happening externally

        if attn_mask is None:
            # Default to causal mask
            attn_mask = self._subsequent_mask(self.sequence_length)

        # Input
        embed_in = lbann.Embedding(x,
                                   num_embeddings=self.vocab_size,
                                   embedding_dim=self.hidden_size,
                                   weights=self.embedding_weights)
        if self.input_dropout > 0:
            embed_in = lbann.Dropout(embed_in,
                                     keep_prob=1 - self.input_dropout,
                                     name=f'input_dropout')

        # Layers
        outputs = embed_in
        for layer in self.layers:
            outputs = layer(outputs, attn_mask)

        # NOTE: Final layer normalization and head are happening externally
        return outputs


class GPTNeoXLayer(lbann.modules.Module):
    """
    GPT-NeoX block.
    
    Consists of either a parallel (default) or sequential MHA/MLP setup.
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 activation: Type[lbann.Layer],
                 intermediate_size: int,
                 layer_norm_eps: float,
                 dropout: float,
                 attn_dropout: float,
                 parallel_mlp: bool,
                 attention_bias: bool,
                 initializer_range: float,
                 out_initializer_range: float,
                 sequence_length: int,
                 positional_encoding: encoding.SequenceEncoding,
                 name: Optional[str] = None):
        self.name = name or ''
        self.dropout = dropout
        self.parallel_mlp = parallel_mlp
        self.sequence_length = sequence_length

        self.input_layernorm = normalization.LayerNorm(
            hidden_size, name=f'{self.name}_input_ln', eps=layer_norm_eps)
        if not self.parallel_mlp:
            self.post_attention_layernorm = normalization.LayerNorm(
                hidden_size,
                name=f'{self.name}_postattn_ln',
                eps=layer_norm_eps)

        self.attention = GPTNeoXAttention(hidden_size, num_heads,
                                          attention_bias, attn_dropout,
                                          initializer_range,
                                          out_initializer_range,
                                          positional_encoding, self.name)
        self.mlp = GPTNeoXMLP(hidden_size,
                              intermediate_size,
                              activation,
                              initializer_range,
                              out_initializer_range,
                              name=self.name)

    def forward(self, input: lbann.Layer, attn_mask: lbann.Layer):
        x = self.input_layernorm(input)
        y = self.attention(x, attn_mask)
        if self.dropout > 0:
            y = lbann.Dropout(y,
                              keep_prob=1 - self.dropout,
                              name=f'{self.name}_dropout')

        if self.parallel_mlp:
            # normed = ln_input(input)
            # result = input + attn(normed) + mlp(normed)
            # NOTE: We use the corrected version of the network from the paper
            #       (where the normed input is tied)
            z = self.mlp(x)
            if self.dropout > 0:
                z = lbann.Dropout(z,
                                  keep_prob=1 - self.dropout,
                                  name=f'{self.name}_mlp_dropout')
            result = lbann.Add(input, y)
            result = lbann.Add(result, z)
        else:
            # result = input + mlp(ln_postattn(input + attn(ln_input(x))))

            # Add residual the first time
            y = lbann.Add(input, y)
            # Use untied layer norm
            y2 = self.post_attention_layernorm(y)

            z = self.mlp(y2)
            if self.dropout > 0:
                z = lbann.Dropout(z,
                                  keep_prob=1 - self.dropout,
                                  name=f'{self.name}_mlp_dropout')
            # Add residual a second time
            result = lbann.Add(input, z)

        return result


class GPTNeoXAttention(lbann.modules.Module):
    """
    Adapter for GPT-NeoX self-attention module to the LBANN built-in multi-head
    attention.
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 attention_bias: bool,
                 attn_dropout: float,
                 initializer_range: float,
                 out_initializer_range: float,
                 positional_encoding: encoding.SequenceEncoding,
                 name: Optional[str] = None):
        self.attn = attention.MultiheadAttention(
            hidden_size,
            num_heads,
            self_attention=True,
            dropout=attn_dropout,
            weight_initializer=(lambda: lbann.NormalInitializer(
                mean=0, standard_deviation=initializer_range)),
            out_weight_initializer=(lambda: lbann.NormalInitializer(
                mean=0, standard_deviation=out_initializer_range)),
            positional_encoding=positional_encoding,
            attn_bias=attention_bias,
            name=name)

    def forward(self, x: lbann.Layer, attn_mask: lbann.Layer):
        return self.attn(x, x, x, mask=attn_mask)


class GPTNeoXMLP(lbann.modules.Module):
    """
    Standard transformer MLP.
    """

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 activation: Type[lbann.Layer],
                 initializer: float,
                 out_initializer: float,
                 name: Optional[str] = None):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.name = name or ''
        self.extra_layer_args = {}
        self.extra_ffn_args = {}

        self.fc1_weights = [
            lbann.Weights(initializer=lbann.NormalInitializer(
                mean=0, standard_deviation=initializer),
                          name=f'{self.name}_fc1_matrix'),
            lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
                          name=f'{self.name}_fc1_bias'),
        ]
        self.fc2_weights = [
            lbann.Weights(initializer=lbann.NormalInitializer(
                mean=0, standard_deviation=out_initializer),
                          name=f'{self.name}_fc2_matrix'),
            lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
                          name=f'{self.name}_fc2_bias'),
        ]

    def forward(self, x):
        y = lbann.ChannelwiseFullyConnected(
            x,
            weights=self.fc1_weights,
            output_channel_dims=[self.intermediate_size],
            name=f'{self.name}_fc1',
            **self.extra_layer_args,
            **self.extra_ffn_args,
        )
        y = self.activation(y,
                            name=f'{self.name}_ffn_act',
                            **self.extra_layer_args,
                            **self.extra_ffn_args)

        y = lbann.ChannelwiseFullyConnected(
            y,
            weights=self.fc2_weights,
            output_channel_dims=[self.hidden_size],
            name=f'{self.name}_fc2',
            **self.extra_layer_args,
            **self.extra_ffn_args,
        )
        return y
