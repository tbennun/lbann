"""
XLNet transformer variant.
"""
from enum import Enum
import lbann
from lbann.modules.transformer import encoding, normalization, attention
import math
import numpy as np
from typing import Optional, Type
import warnings


class XLAttentionType(Enum):
    bi = 0  # XLNet
    uni = 1  # Transformer-XL


class XLNet(lbann.modules.Module):
    """
    Implements an XLNet transformer variant for permutation language modeling.

    For more information, see Z. Yang et al., "XLNet: Generalized Autoregressive
    Pretraining for Language Understanding," NeurIPS 2019.

    This is a port of the HuggingFace implementation of XLNet. The original
    version can be found at:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/modeling_xlnet.py
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 1024,
        n_layer: int = 24,
        n_head: int = 15,
        d_inner: int = 4096,
        activation: Type[lbann.Layer] = lbann.Gelu,
        untie_r: bool = True,  # Whether to untie relative position biases
        attn_type: XLAttentionType = XLAttentionType.bi,
        sequence_length: int = 512,
        # Attention parameters
        use_mems_train: bool = False,  # Use recurrent memory mechanism
        clamp_len: int = -1,  # Relative positional encoding clamp length
        same_length: bool = False,  # Use same attn. length for each token
        bi_data: bool = False,  # Bidirectional input (True for pretraining)
        # Dropout
        dropout: float = 0.1,
        # Positional encoding
        positional_encoding: Optional[encoding.SequenceEncoding] = None,
        # Initialization
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = d_model
        self.num_heads = n_head
        self.intermediate_size = d_inner
        self.input_dropout = dropout
        self.sequence_length = sequence_length
        self.separate_heads = False
        self.attn_type = attn_type
        self.bi_data = bi_data
        self.clamp_len = clamp_len
        self.name = 'xlnet'

        assert not use_mems_train

        if positional_encoding is not None:
            raise TypeError(
                'XLNet was designed to train with its own positional encoder')

        self.layers = [
            XLNetLayer(d_model, n_head, activation, d_inner, layer_norm_eps,
                       dropout, initializer_range, initializer_range,
                       sequence_length, positional_encoding, f'layer_{i}')
            for i in range(n_layer)
        ]

        # Some temporary buffers
        self.eye_qlen = -np.eye(sequence_length)
        self.cached_positional_encoding = {}

    def forward(self,
                x: lbann.Layer,
                input_mask: Optional[lbann.Layer] = None,
                perm_mask: Optional[lbann.Layer] = None,
                target_mapping: Optional[lbann.Layer] = None):
        # NOTE: Embedding and positional encoding is happening externally
        #bsz, qlen = x.shape[0], x.shape[1]
        bsz, qlen = None, self.sequence_length
        klen = qlen

        if self.attn_type == XLAttentionType.uni:
            raise NotImplementedError(
                'Unidirectional attention is currently unsupported')
            # mlen = 0
            # attn_mask = self.create_uni_mask(qlen, mlen)
        elif self.attn_type == XLAttentionType.bi:
            attn_mask = None

        # TODO: Get input, input_mask, perm_mask, and target_mapping from input
        # Create mask from permutation and input mask (if any exist)
        if input_mask is not None and perm_mask is not None:
            input_mask = lbann.Add(input_mask, perm_mask)
        elif input_mask is None and perm_mask is not None:
            input_mask = perm_mask
        else:
            input_mask = None

        if input_mask is not None:
            if attn_mask is None:
                attn_mask = input_mask
            else:
                attn_mask = lbann.Add(attn_mask, input_mask)

        if attn_mask is not None:
            non_tgt_mask = lbann.Add(attn_mask, self.neye_qlen)
        else:
            non_tgt_mask = None

        word_emb_k = x
        if self.input_dropout > 0:
            output_h = lbann.Dropout(word_emb_k,
                                     keep_prob=1 - self.input_dropout)
        else:
            output_h = word_emb_k

        if target_mapping is not None:
            word_emb_q = self.mask_emb
            if self.input_dropout > 0:
                output_g = lbann.Dropout(word_emb_q,
                                         keep_prob=1 - self.input_dropout)
            else:
                output_g = word_emb_q
        else:
            output_g = None

        pos_emb = self.relative_positional_encoding(qlen, klen)
        if self.input_dropout > 0:
            pos_emb = lbann.Dropout(pos_emb, keep_prob=1 - self.input_dropout)

        # Layers
        for layer in self.layers:
            output_h, output_g = layer(
                output_h,
                output_g,
                attn_mask_h=non_tgt_mask,
                attn_mask_g=attn_mask,
                r=pos_emb,
                target_mapping=target_mapping,
            )

        outputs = output_g if output_g is not None else output_h
        if self.input_dropout > 0:
            outputs = lbann.Dropout(outputs, keep_prob=1 - self.input_dropout)

        # NOTE: Final layer normalization and head are happening externally
        return outputs

    # Implementation of XLNet variant of relative positional encoding
    # adapted from the HuggingFace implementation.
    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        # NOTE: This function only works on numpy arrays, so no need for LBANN
        # layers.
        sinusoid_inp = np.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = np.concatenate(
            [np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1)
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = np.broadcast_to(
                pos_emb, (pos_emb.shape[0], bsz, pos_emb.shape[2]))

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        if (qlen, klen) in self.cached_positional_encoding:
            return self.cached_positional_encoding[(qlen, klen)]

        def _clamp(x, minval, maxval):
            return np.minimum(np.maximum(x, minval), maxval)

        # create relative positional encoding.
        freq_seq = np.arange(0, self.hidden_size, 2.0,
                             dtype=np.int64).astype(np.float32)
        inv_freq = 1 / np.pow(10000, (freq_seq / self.hidden_size))

        if self.attn_type == XLAttentionType.bi:
            beg, end = klen, -qlen
        elif self.attn_type == XLAttentionType.uni:
            beg, end = klen, -1

        if self.bi_data:
            fwd_pos_seq = np.arange(beg, end, -1.0,
                                    dtype=np.int64).astype(np.float32)
            bwd_pos_seq = np.arange(-beg, -end, 1.0,
                                    dtype=np.int64).astype(np.float32)

            if self.clamp_len > 0:

                fwd_pos_seq = _clamp(fwd_pos_seq, -self.clamp_len,
                                     self.clamp_len)
                bwd_pos_seq = _clamp(bwd_pos_seq, -self.clamp_len,
                                     self.clamp_len)

            if bsz is not None:
                fwd_pos_emb = self.positional_embedding(
                    fwd_pos_seq, inv_freq, bsz // 2)
                bwd_pos_emb = self.positional_embedding(
                    bwd_pos_seq, inv_freq, bsz // 2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            pos_emb = np.concatenate([fwd_pos_emb, bwd_pos_emb], axis=1)
        else:
            fwd_pos_seq = np.arange(beg, end, -1.0,
                                    dtype=np.int64).astype(np.float32)
            if self.clamp_len > 0:
                fwd_pos_seq = _clamp(fwd_pos_seq, -self.clamp_len,
                                     self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        self.cached_positional_encoding[(qlen, klen)] = pos_emb
        return pos_emb


class XLNetLayer(lbann.modules.Module):
    """
    XLNet block.
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 activation: Type[lbann.Layer],
                 intermediate_size: int,
                 layer_norm_eps: float,
                 dropout: float,
                 initializer_range: float,
                 out_initializer_range: float,
                 sequence_length: int,
                 positional_encoding: encoding.SequenceEncoding,
                 name: Optional[str] = None):
        self.name = name or ''
        self.dropout = dropout
        self.sequence_length = sequence_length

        self.attention = XLNetAttention(hidden_size, num_heads,
                                        initializer_range,
                                        out_initializer_range,
                                        positional_encoding, self.name)
        self.mlp = XLNetMLP(hidden_size,
                            intermediate_size,
                            activation,
                            initializer_range,
                            out_initializer_range,
                            layer_norm_eps,
                            name=self.name)

    def forward(self,
                h: lbann.Layer,
                g: lbann.Layer,
                attn_mask_h: Optional[lbann.Layer],
                attn_mask_g: Optional[lbann.Layer],
                r: lbann.Layer,
                target_mapping: Optional[lbann.Layer] = None):
        output_h, output_g = self.attention(h,
                                            g,
                                            attn_mask_h,
                                            attn_mask_g,
                                            r,
                                            target_mapping=target_mapping)

        if g is not None:
            output_g = self.mlp(output_g)
        output_h = self.mlp(output_h)

        return output_h, output_g


class XLNetAttention(lbann.modules.Module):
    """
    Implementation of the XLNet relative attention module.
    """
    global_count = 0  # Static counter, used for default names

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 initializer_range: float,
                 out_initializer_range: float,
                 dropout: float,
                 name: Optional[str] = None):
        XLNetAttention.global_count += 1
        self.instance = 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1 / math.sqrt(hidden_size)
        self.dropout = dropout

        # Module name
        self.name = name
        if not self.name:
            self.name = f'xlattn{XLNetAttention.global_count}'

        weight_initializer = lambda: lbann.NormalInitializer(
            mean=0, standard_deviation=initializer_range)
        out_weight_initializer = lambda: lbann.NormalInitializer(
            mean=0, standard_deviation=out_initializer_range)
        bias_initializer = lambda: lbann.ConstantInitializer(value=0)

        self.query_weights = [
            lbann.Weights(initializer=weight_initializer(),
                          name=f'{self.name}_query_matrix'),
        ]
        self.key_weights = [
            lbann.Weights(initializer=weight_initializer(),
                          name=f'{self.name}_key_matrix'),
        ]
        self.value_weights = [
            lbann.Weights(initializer=weight_initializer(),
                          name=f'{self.name}_value_matrix'),
        ]
        self.r_weights = [
            lbann.Weights(initializer=weight_initializer(),
                          name=f'{self.name}_r_matrix'),
            lbann.Weights(initializer=bias_initializer(),
                          name=f'{self.name}_r_r_bias'),
            lbann.Weights(initializer=bias_initializer(),
                          name=f'{self.name}_r_s_bias'),
            lbann.Weights(initializer=bias_initializer(),
                          name=f'{self.name}_r_w_bias'),
        ]
        self.output_weights = [
            lbann.Weights(initializer=out_weight_initializer(),
                          name=f'{self.name}_output_matrix'),
        ]

        self.norm = normalization.LayerNorm(hidden_size,
                                            name=f'{self.name}_norm')

    def forward(self,
                h: lbann.Layer,
                g: Optional[lbann.Layer],
                attn_mask_h: Optional[lbann.Layer],
                attn_mask_g: Optional[lbann.Layer],
                r: lbann.Layer,
                target_mapping: Optional[lbann.Layer] = None,
                **extra_kwargs):
        self.instance += 1
        name = f'{self.name}_instance{self.instance}'

        keys_fc = lbann.ChannelwiseFullyConnected(
            h,
            weights=self.key_weights,
            output_channel_dims=[self.embed_dim],
            bias=False,
            name=f'{name}_keys_fc',
            **extra_kwargs,
        )
        values_fc = lbann.ChannelwiseFullyConnected(
            h,
            weights=self.value_weights,
            output_channel_dims=[self.embed_dim],
            bias=False,
            name=f'{name}_values_fc',
            **extra_kwargs,
        )
        # This would use ``mems || h``
        queries_fc = lbann.ChannelwiseFullyConnected(
            h,
            weights=self.query_weights,
            output_channel_dims=[self.embed_dim],
            bias=False,
            name=f'{name}_queries_fc',
            **extra_kwargs,
        )

        # Learned positional encoding
        k_head_r = lbann.ChannelwiseFullyConnected(
            r,
            weights=[self.r_weights[0]],
            output_channel_dims=[self.embed_dim],
            bias=False,
            name=f'{name}_queries_fc',
            **extra_kwargs,
        )

        h_attentions = self.rel_attn(queries_fc,
                                     keys_fc,
                                     values_fc,
                                     k_head_r,
                                     attn_mask=attn_mask_h,
                                     name=name,
                                     **extra_kwargs)
        output_h = self.post_attention(h, h_attentions, name, **extra_kwargs)

        if g is not None:  # Two-stream attention
            q_head_g = lbann.ChannelwiseFullyConnected(
                g,
                weights=self.query_weights,
                output_channel_dims=[self.embed_dim],
                bias=False,
                name=f'{name}_g_queries_fc',
                **extra_kwargs,
            )
            if target_mapping is not None:
                # TODO: Some einsum...
                # HxPxS * BxHxPxS -> BxHxPxS
                q_head_g = lbann.MatMul(
                    q_head_g,
                    target_mapping,
                    transpose_a=False,
                    transpose_b=False,
                    name=f'{name}_qg_times_mapping',
                    **extra_kwargs,
                )

            g_attentions = self.rel_attn(q_head_g,
                                         keys_fc,
                                         values_fc,
                                         k_head_r,
                                         attn_mask=attn_mask_g,
                                         name=name,
                                         **extra_kwargs)

            if target_mapping is not None:
                # BxHxPxS * BxHxPxS -> BxHxPxS
                g_attentions = lbann.MatMul(
                    q_head_g,
                    target_mapping,
                    transpose_a=False,
                    transpose_b=False,
                    name=f'{name}_gattn_times_mapping',
                    **extra_kwargs,
                )

            output_g = self.post_attention(g, g_attentions, name,
                                           **extra_kwargs)
        else:
            output_g = None

        return output_h, output_g

    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        x_size = x.shape

        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        x = x[:, :, :, :klen]

        return x

    def rel_attn(self,
                 q: lbann.Layer,
                 k: lbann.Layer,
                 v: lbann.Layer,
                 r: lbann.Layer,
                 attn_mask: Optional[lbann.Layer] = None,
                 name: str = '',
                 **extra_kwargs):

        # TODO: TensorPermute, reshape
        ac = lbann.MatMul(
            lbann.Add(q, self.r_weights[3]),
            k,
            transpose_a=True,
            transpose_b=False,
            name=f'{name}_qk',
            **extra_kwargs,
        )
        bd = lbann.MatMul(
            lbann.Add(q, self.r_weights[1]),
            r,
            transpose_a=True,
            transpose_b=False,
            name=f'{name}_posattn',
            **extra_kwargs,
        )
        bd = self.rel_shift_bnij(bd, klen=self.sequence_length)

        attn_score = lbann.Scale(lbann.Add(ac, bd), constant=self.scale)
        if attn_mask is not None:
            # attn_score = attn_score - 1e30 * attn_mask
            attn_score = lbann.Add(attn_score,
                                   attn_mask,
                                   name=f'{name}_mask',
                                   **extra_kwargs)

        attn_prob = lbann.ChannelwiseSoftmax(attn_score,
                                             dim=-1,
                                             single_dim_mode=True,
                                             name=f'{name}_softmax',
                                             **extra_kwargs)
        if self.dropout > 0:
            attn_prob = lbann.Dropout(
                attn_prob,
                keep_prob=1 - self.dropout,
                name=f'{name}_drop',
                **extra_kwargs,
            )

        # Attention output as batched matrix multiplication
        # HxSxS * HxSxP -> HxSxP
        attentions = lbann.MatMul(attn_prob,
                                  v,
                                  transpose_b=True,
                                  name=f'{name}_vmult',
                                  **extra_kwargs)
        # TODO: TensorPermute, reshape
        return attentions

    def post_attention(self,
                       h,
                       attentions,
                       name,
                       residual=True,
                       **extra_kwargs):
        # Post-attention processing
        outputs_fc = lbann.ChannelwiseFullyConnected(
            attentions,
            weights=self.output_weights,
            output_channel_dims=[self.embed_dim],
            name=f'{name}_postattn',
            **extra_kwargs,
        )

        if self.dropout > 0:
            outputs = lbann.Dropout(outputs_fc, keep_prob=1 - self.dropout)
        else:
            outputs = outputs_fc

        if residual:
            outputs = lbann.Add(outputs, h)
        outputs = self.norm(outputs)
        return outputs


class XLNetMLP(lbann.modules.Module):
    """
    Standard transformer MLP.
    """

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 activation: Type[lbann.Layer],
                 initializer: float,
                 out_initializer: float,
                 layer_norm_eps: float,
                 name: Optional[str] = None):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.extra_layer_args = {}
        self.extra_ffn_args = {}

        # Module name
        self.name = name
        if not self.name:
            self.name = f'xlmlp{XLNetAttention.global_count}'

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

        self.ln = normalization.LayerNorm(hidden_size,
                                          eps=layer_norm_eps,
                                          name=f'{self.name}_ffn_ln')

    def forward(self, x):
        self.instance += 1
        name = f'{self.name}_instance{self.instance}'

        y = lbann.ChannelwiseFullyConnected(
            x,
            weights=self.fc1_weights,
            output_channel_dims=[self.intermediate_size],
            name=f'{name}_fc1',
            **self.extra_layer_args,
            **self.extra_ffn_args,
        )
        y = self.activation(y,
                            name=f'{name}_ffn_act',
                            **self.extra_layer_args,
                            **self.extra_ffn_args)
        if self.dropout > 0:
            y = lbann.Dropout(y,
                              keep_prob=1 - self.dropout,
                              name=f'{name}_mlp_dropout_1')
        y = lbann.ChannelwiseFullyConnected(
            y,
            weights=self.fc2_weights,
            output_channel_dims=[self.hidden_size],
            name=f'{name}_fc2',
            **self.extra_layer_args,
            **self.extra_ffn_args,
        )
        if self.dropout > 0:
            y = lbann.Dropout(y,
                              keep_prob=1 - self.dropout,
                              name=f'{name}_mlp_dropout_2')
        z = lbann.Add(y, x)
        return self.ln(z)
