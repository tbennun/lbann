import argparse
from enum import Enum, auto
import lbann
import math
from typing import Optional, Tuple
from lbann.models.transformer import Transformer
import lbann.modules
from lbann.modules.transformer import encoding
from lbann.modules.transformer.normalization import LayerNorm
import parallelism


class InputEncoding(Enum):
    """ Different types of input encoding used by the transformer samples. """
    POSITIONAL = auto()  # Positional encoding
    LEARNED = auto()  # Learned embeddings
    ROPE = auto()  # Rotary positional embedding
    NONE = auto()  # No encoding


def create_encoder_decoder_transformer(dataset, args: argparse.Namespace):
    """
    Creates an Encoder-Decoder Transformer model as per Vaswani et al., 
    "Attention is all you need" (2017), with a language modeling head for
    sequence transduction tasks (e.g. translation).
    """
    num_encoders = num_decoders = args.num_layers
    sequence_length = dataset.sequence_length
    vocab_size = dataset.vocab_size()

    # Embedding weights
    var = 2 / (args.embed_dim + vocab_size)  # Glorot initialization
    embedding_weights = lbann.Weights(
        name='embeddings',
        initializer=lbann.NormalInitializer(standard_deviation=math.sqrt(var)),
    )

    # Input is two sequences of token IDs
    input_tokens = lbann.Input(data_field='samples')

    # Get sequences of embedding vectors
    # Note: Scale embeddings by sqrt(embed_dim).
    # Note: Decoder input is shifted right, so embedding for last
    # token isn't needed.
    embeddings_tokens = lbann.Identity(
        lbann.Slice(
            input_tokens,
            axis=0,
            slice_points=[0, 2 * sequence_length - 1],
        ))
    embeddings = lbann.Embedding(
        embeddings_tokens,
        weights=embedding_weights,
        num_embeddings=vocab_size,
        embedding_dim=args.embed_dim,
        padding_idx=dataset.pad_index,
    )
    embeddings = lbann.WeightedSum(
        embeddings,
        scaling_factors=math.sqrt(args.embed_dim),
    )
    embeddings_slice = lbann.Slice(
        embeddings,
        axis=0,
        slice_points=[0, sequence_length, 2 * sequence_length - 1],
    )
    encoder_input = lbann.Identity(embeddings_slice)
    decoder_input = lbann.Identity(embeddings_slice)
    petype = InputEncoding[args.positional_encoding.upper()]

    # Apply input encoding
    encoder_input, decoder_input, posenc = _add_input_encoding(
        encoder_input, decoder_input, petype, args.embed_dim,
        args.input_dropout, sequence_length, sequence_length - 1,
        args.num_attention_heads, args.rope_ratio)

    # Add encoder-decoder transformer model
    transformer = Transformer(hidden_size=args.embed_dim,
                              num_heads=args.num_attention_heads,
                              dropout=args.dropout,
                              feedforward_size=args.feedforward_dim,
                              name='transformer',
                              num_encoder_layers=num_encoders,
                              num_decoder_layers=num_decoders,
                              positional_encoding=posenc)

    # Apply parallelism techniques
    transformer, extra_model_kwargs = parallelism.apply_subgraph_parallelism(
        transformer, args)
    parallelism.apply_ffn_model_parallelism(transformer, args)
    parallelism.apply_fsdp_mlp(transformer, [embedding_weights], args)
    parallelism.apply_layer_parallelism(transformer, args)

    # Run through transformer
    result = transformer(encoder_input, decoder_input, sequence_length - 1)

    # Reconstruct decoder input
    lm_head = lbann.ChannelwiseFullyConnected(result,
                                              weights=embedding_weights,
                                              output_channel_dims=[vocab_size],
                                              bias=False,
                                              transpose=True,
                                              name="prediction_layer")
    preds = lbann.ChannelwiseSoftmax(lm_head)
    preds = lbann.TensorPermute(preds, axes=[1, 0])

    # Compute loss
    loss = _add_encoder_decoder_loss(preds, input_tokens, sequence_length,
                                     vocab_size, dataset.pad_index)

    parallelism.apply_lm_head_model_parallelism(lm_head, args)

    # Construct model
    metrics = []
    callbacks = [
        lbann.CallbackPrint(),
        lbann.CallbackTimer(),
        lbann.CallbackGPUMemoryUsage()
    ]
    result = lbann.Model(
        args.num_epochs,
        layers=lbann.traverse_layer_graph(input_tokens),
        objective_function=loss,
        metrics=metrics,
        callbacks=callbacks,
        **extra_model_kwargs,
    )

    parallelism.apply_fsdp_allweights(result, args)
    parallelism.apply_layer_parallelism_postamble(result, args)
    return result


def create_causal_lm_decoder_transformer(
        dataset,
        embed_dim: int,
        num_decoders: int,
        num_heads: int,
        dropout: float,
        input_dropout: float,
        attn_dropout: float,
        num_epochs: int,
        args: argparse.Namespace,
        transformer: Optional[lbann.modules.Module] = None,
        classifier_dropout: float = 0.0):
    """
    Creates a GPT-style decoder-only transformer for causal language modeling
    tasks (e.g., predict next token).
    """
    sequence_length = dataset.sequence_length
    vocab_size = dataset.vocab_size()

    # Embedding weights
    var = 2 / (embed_dim + vocab_size)  # Glorot initialization
    embedding_weights = lbann.Weights(
        name='embeddings',
        initializer=lbann.NormalInitializer(standard_deviation=math.sqrt(var)),
    )

    # Input is a sequences of token IDs
    input_tokens = lbann.Input(data_field='samples')

    # Get sequences of embedding vectors
    embeddings = lbann.Embedding(
        input_tokens,
        weights=embedding_weights,
        num_embeddings=vocab_size,
        embedding_dim=embed_dim,
        padding_idx=dataset.pad_index,
    )
    decoder_input = lbann.WeightedSum(
        embeddings,
        scaling_factors=math.sqrt(embed_dim),
    )

    petype = InputEncoding[args.positional_encoding.upper()]

    # Apply input encoding
    _, decoder_input, posenc = _add_input_encoding(None, decoder_input, petype,
                                                   embed_dim, input_dropout, 0,
                                                   sequence_length, num_heads,
                                                   args.rope_ratio)

    # Add a GPT-style (decoder-only) transformer model
    if transformer is None:
        use_transformer_params = True
        transformer = Transformer(hidden_size=embed_dim,
                                  num_heads=num_heads,
                                  dropout=dropout,
                                  attn_dropout=attn_dropout,
                                  num_encoder_layers=0,
                                  num_decoder_layers=num_decoders,
                                  pre_layernorm=True,
                                  activation=lbann.Gelu,
                                  positional_encoding=posenc,
                                  name='transformer')
    else:
        use_transformer_params = False

    # Apply parallelism techniques
    transformer, extra_model_kwargs = parallelism.apply_subgraph_parallelism(
        transformer, args)
    parallelism.apply_ffn_model_parallelism(transformer, args)
    parallelism.apply_fsdp_mlp(transformer, [embedding_weights], args)
    parallelism.apply_layer_parallelism(transformer, args)

    # Run through transformer with the same sequence
    if use_transformer_params:
        # Encoder-decoder parameters
        result = transformer(decoder_input, decoder_input, sequence_length)
    else:
        # Decoder-only parameters
        # TODO: Feed in attention mask
        result = transformer(decoder_input, None)

    # Apply layer normalization on the outputs
    norm_final = LayerNorm(embed_dim, name=f'final_layernorm')
    result = norm_final(result)

    # Apply classifier dropout, if exists
    if classifier_dropout > 0:
        result = lbann.Dropout(result,
                               keep_prob=1 - classifier_dropout,
                               name=f'classifier_dropout')

    # Apply language modeling head on results
    lm_head = lbann.ChannelwiseFullyConnected(result,
                                              weights=embedding_weights,
                                              output_channel_dims=[vocab_size],
                                              bias=False,
                                              transpose=True,
                                              name="prediction_layer")
    preds = lbann.ChannelwiseSoftmax(lm_head)
    preds = lbann.TensorPermute(preds, axes=[1, 0])

    # Compute loss
    loss = _add_autoregressive_loss(preds, input_tokens, sequence_length,
                                    vocab_size, dataset.pad_index)

    parallelism.apply_lm_head_model_parallelism(lm_head, args)

    # Construct model
    metrics = []
    callbacks = [
        lbann.CallbackPrint(),
        lbann.CallbackTimer(),
        lbann.CallbackGPUMemoryUsage()
    ]
    result = lbann.Model(
        num_epochs,
        layers=lbann.traverse_layer_graph(input_tokens),
        objective_function=loss,
        metrics=metrics,
        callbacks=callbacks,
        **extra_model_kwargs,
    )

    parallelism.apply_fsdp_allweights(result, args)
    parallelism.apply_layer_parallelism_postamble(result, args)
    return result


def create_masked_language_modeling_transformer(
        dataset,
        embed_dim: int,
        num_encoders: int,
        num_decoders: int,
        num_heads: int,
        dropout: float,
        input_dropout: float,
        attn_dropout: float,
        num_epochs: int,
        args: argparse.Namespace,
        transformer: Optional[lbann.modules.Module] = None):
    """
    Creates a flexible transformer for masked language modeling tasks.
    """
    sequence_length = dataset.sequence_length
    vocab_size = dataset.vocab_size()

    # Embedding weights
    var = 2 / (embed_dim + vocab_size)  # Glorot initialization
    embedding_weights = lbann.Weights(
        name='embeddings',
        initializer=lbann.NormalInitializer(standard_deviation=math.sqrt(var)),
    )

    # Input is a sequences of token IDs followed by a mask sequence
    all_inputs = lbann.Input(data_field='samples')
    slice_points = [
        0,
        sequence_length,  # Original sequence
        2 * sequence_length,  # Mask
    ]
    if args.attn_mask:
        # Attention matrix mask
        slice_points.append(2 * sequence_length +
                            sequence_length * sequence_length)

    slc = lbann.Slice(
        all_inputs,
        slice_points=slice_points,
    )
    input_tokens = lbann.Identity(slc)
    mask = lbann.Identity(slc)
    if args.attn_mask:
        attn = lbann.Reshape(lbann.Identity(slc),
                             dims=[sequence_length, sequence_length])
    else:
        attn = None

    masked_input = lbann.Select(mask,
                                input_tokens,
                                value=1,
                                if_false=dataset.mask_index)

    # Get sequences of embedding vectors
    embeddings = lbann.Embedding(
        masked_input,
        weights=embedding_weights,
        num_embeddings=vocab_size,
        embedding_dim=embed_dim,
        padding_idx=dataset.pad_index,
    )
    decoder_input = lbann.WeightedSum(
        embeddings,
        scaling_factors=math.sqrt(embed_dim),
    )

    petype = InputEncoding[args.positional_encoding.upper()]

    # Apply input encoding
    _, decoder_input, posenc = _add_input_encoding(None, decoder_input, petype,
                                                   embed_dim, input_dropout, 0,
                                                   sequence_length, num_heads)

    if transformer is None:
        use_transformer_params = True
        transformer = Transformer(hidden_size=embed_dim,
                                  num_heads=num_heads,
                                  dropout=dropout,
                                  attn_dropout=attn_dropout,
                                  num_encoder_layers=num_encoders,
                                  num_decoder_layers=num_decoders,
                                  pre_layernorm=True,
                                  activation=lbann.Gelu,
                                  positional_encoding=posenc,
                                  name='transformer')
    else:
        use_transformer_params = False

    # Tessellate attention pattern for all heads (note that this is a memory issue)
    if attn is not None and not transformer.separate_heads:
        # TODO(later): Use broadcasting semantics to save memory
        attn = lbann.Reshape(attn, dims=[1, sequence_length, sequence_length])
        attn = lbann.Tessellate(
            attn, dims=[num_heads, sequence_length, sequence_length])

    # Apply parallelism techniques
    transformer, extra_model_kwargs = parallelism.apply_subgraph_parallelism(
        transformer, args)
    parallelism.apply_ffn_model_parallelism(transformer, args)
    parallelism.apply_fsdp_mlp(transformer, [embedding_weights], args)
    parallelism.apply_layer_parallelism(transformer, args)

    # Run through transformer with the same sequence
    if use_transformer_params:
        # Encoder-decoder parameters
        result = transformer(decoder_input,
                             decoder_input,
                             sequence_length,
                             target_mask=attn)
    else:
        # Decoder-only parameters
        result = transformer(decoder_input, attn)

    # Apply layer normalization on the outputs
    norm_final = LayerNorm(embed_dim, name=f'final_layernorm')
    result = norm_final(result)

    # Apply language modeling head on results
    lm_head = lbann.ChannelwiseFullyConnected(result,
                                              weights=embedding_weights,
                                              output_channel_dims=[vocab_size],
                                              bias=False,
                                              transpose=True,
                                              name="prediction_layer")
    preds = lbann.ChannelwiseSoftmax(lm_head)
    preds = lbann.TensorPermute(preds, axes=[1, 0])

    # Compute loss
    loss = _add_mlm_loss(preds, input_tokens, sequence_length, vocab_size,
                         dataset.pad_index)

    parallelism.apply_lm_head_model_parallelism(lm_head, args)

    # Construct model
    metrics = []
    callbacks = [
        lbann.CallbackPrint(),
        lbann.CallbackTimer(),
        lbann.CallbackGPUMemoryUsage()
    ]
    result = lbann.Model(
        num_epochs,
        layers=lbann.traverse_layer_graph(input_tokens),
        objective_function=loss,
        metrics=metrics,
        callbacks=callbacks,
        **extra_model_kwargs,
    )

    parallelism.apply_fsdp_allweights(result, args)
    parallelism.apply_layer_parallelism_postamble(result, args)
    return result


def _add_input_encoding(
    encoder_input: lbann.Layer, decoder_input: lbann.Layer,
    encoding_kind: InputEncoding, embed_dim: int, input_dropout: float,
    encoder_sequence_length: int, decoder_sequence_length: int, num_heads: int,
    rope_ratio: float
) -> Tuple[lbann.Layer, lbann.Layer, encoding.SequenceEncoding]:
    if encoding_kind == InputEncoding.NONE:
        # Do nothing
        return encoder_input, decoder_input, None

    elif encoding_kind == InputEncoding.POSITIONAL:
        # Trigonometric positional encoding
        positional_encoder = encoding.PositionalEncoding(
            embed_dim, input_dropout)
        kwargs = {}
    elif encoding_kind == InputEncoding.LEARNED:
        # Learned (embedding) encoding
        max_seqlen = max(encoder_sequence_length, decoder_sequence_length)
        positional_encoder = encoding.LearnedInputEncoding(
            embed_dim, max_seqlen, input_dropout)
        # Optimize by not computing embeddings twice
        kwargs = dict(learned_encoding=positional_encoder.compute_embeddings())
    elif encoding_kind == InputEncoding.ROPE:
        freq_dim = (embed_dim // num_heads) * rope_ratio
        max_seqlen = max(encoder_sequence_length, decoder_sequence_length)
        positional_encoder = encoding.RotaryPositionalEmbedding(
            freq_dim, max_seqlen, num_heads)
        kwargs = {}
    else:
        raise TypeError(
            f'Unsupported positional encoder type: {encoding_kind}')

    # Apply encoder
    if encoder_input is not None:
        encoder_input = positional_encoder.apply_input(
            encoder_input, encoder_sequence_length, **kwargs)
    if decoder_input is not None:
        decoder_input = positional_encoder.apply_input(
            decoder_input, decoder_sequence_length, **kwargs)

    return encoder_input, decoder_input, positional_encoder


def _add_encoder_decoder_loss(preds, both_sequences, sequence_length,
                              vocab_size, pad_index):
    # Get label tokens from the second sequence, starting from the second token
    # onwards
    target_sequence = lbann.Identity(
        lbann.Slice(
            both_sequences,
            slice_points=[sequence_length + 1, 2 * sequence_length],
        ))
    labels = lbann.Reshape(target_sequence, dims=[1, sequence_length - 1])

    # Filter out output predictions that are in padding from cross-entropy by
    # using values that will never contribute to the cross-entropy loss
    labels = lbann.Select(labels,
                          lbann.Identity(labels),
                          value=pad_index,
                          if_true=(vocab_size + 1))

    # Compute cross-entropy
    ce = lbann.CrossEntropy(preds, labels, use_labels=True)
    return lbann.Scale(ce, constant=1 / (sequence_length - 1))


def _add_autoregressive_loss(preds, input_tokens, sequence_length, vocab_size,
                             pad_index):
    # Compute cross-entropy loss between preds[:-1] (up until the last token)
    # and input[1:] (predicting one token forward)
    shifted_preds = lbann.Identity(
        lbann.Slice(preds, axis=1, slice_points=[0, sequence_length - 1]))
    shifted_labels = lbann.Identity(
        lbann.Slice(input_tokens, slice_points=[1, sequence_length]))

    # Flatten labels
    flat_labels = lbann.Reshape(shifted_labels, dims=[1, sequence_length - 1])

    # Filter out output predictions that are in padding from cross-entropy by
    # using values that will never contribute to the cross-entropy loss
    flat_labels = lbann.Select(flat_labels,
                               lbann.Identity(flat_labels),
                               value=pad_index,
                               if_true=(vocab_size + 1))

    # Compute mean cross-entropy over the sequence
    ce = lbann.CrossEntropy(shifted_preds, flat_labels, use_labels=True)
    return lbann.Scale(ce, constant=1 / (sequence_length - 1))


def _add_mlm_loss(preds, input_tokens, sequence_length, vocab_size, pad_index):
    # Compute cross-entropy loss between preds and the original tokens from a
    # masked input

    # Flatten labels
    flat_labels = lbann.Reshape(input_tokens, dims=[1, sequence_length])

    # Filter out output predictions that are in padding from cross-entropy by
    # using values that will never contribute to the cross-entropy loss
    flat_labels = lbann.Select(flat_labels,
                               lbann.Identity(flat_labels),
                               value=pad_index,
                               if_true=(vocab_size + 1))

    # Compute mean cross-entropy over the sequence
    ce = lbann.CrossEntropy(preds, flat_labels, use_labels=True)
    return lbann.Scale(ce, constant=1 / sequence_length)


# Command-line arguments
def add_transformer_architecture_arguments(args: argparse.Namespace):
    """
    Adds the command line arguments to specify transformer architecture model
    parameters. This is only relevant for the encoder-decoder transformer model.
    """
    args.add_argument('--num-attention-heads',
                      action='store',
                      default=8,
                      type=int,
                      help='number of parallel attention layers (default: 8)',
                      metavar='NUM')
    args.add_argument('--embed-dim',
                      action='store',
                      default=512,
                      type=int,
                      help='embedding space dimension (default: 512)',
                      metavar='NUM')
    args.add_argument('--feedforward-dim',
                      action='store',
                      default=0,
                      type=int,
                      help='feedforward network dimension. If zero, set to be '
                      '4 times the embedding dimension (default: 0)',
                      metavar='NUM')
    args.add_argument('--num-layers',
                      action='store',
                      default=6,
                      type=int,
                      help='Number of encoder and decoder layers (default: 6)',
                      metavar='NUM')
    args.add_argument('--positional-encoding',
                      type=str,
                      default='learned',
                      help='The type of positional encoding to use '
                      '(default: learned)',
                      choices=[s.name.lower() for s in InputEncoding])
    args.add_argument('--rope-ratio',
                      type=float,
                      default=1.0,
                      help='Rotary Positional Embedding ratio (default: 1)')
