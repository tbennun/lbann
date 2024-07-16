"""
Driver script for LBANN pre-training an XLNet model with a permutation language
modeling target. The Pile dataset reader is provided as an example task.
"""
import argparse
from dataclasses import dataclass
from typing import Optional
import math
import os
import os.path
import sys

import lbann
import lbann.contrib.args
from lbann.launcher.batch_script import BatchScript
import lbann.util.amp
from lbann.modules.transformer import encoding
from lbann.models.transformer.xlnet import XLNet

# Local imports
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import dataset_utils
import modeling
import parallelism
import trainer


def main():
    # Setup command line options
    parser = argparse.ArgumentParser()
    lbann.contrib.args.add_scheduler_arguments(parser, 'xlnet')
    lbann.contrib.args.add_profiling_arguments(parser)
    lbann.contrib.args.add_training_arguments(parser,
                                              default_minibatch_size=16)
    lbann.contrib.args.add_amp_arguments(parser)
    parallelism.add_transformer_parallelism_arguments(parser)
    trainer.add_training_arguments(parser)
    dataset_utils.add_dataset_arguments(parser, default='thepile_mlm')

    parser.add_argument('--optimizer',
                        type=str,
                        default='adamw',
                        choices=['adam', 'adamw'],
                        help='Stochastic optimizer used in training')

    # Model parameters
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout ratio in transformer model. 0 disables (default: 0.1)")
    parser.add_argument("--grad-clip",
                        type=float,
                        default=1.0,
                        help="Clip global gradient norm (default: 1.0)")
    parser.add_argument(
        "--fractional-schedule",
        action='store_true',
        default=False,
        help="Use dataset fraction to determine hyperparameter schedule"
        " (default: False)")

    parser.set_defaults(progress=True, num_epochs=1, dataset='thepile_mlm')
    args = parser.parse_args()

    # Load dataset
    os.environ['THE_PILE_SEQUENCE_LENGTH'] = '2048'
    dataset = dataset_utils.load_dataset(args.dataset)

    # Construct transformer
    print('Vocabulary size:', dataset.vocab_size())
    transformer = XLNet(dataset.vocab_size())
    args.positional_encoding = 'none'  # Handled inside model    

    # Construct model around the transformer
    model: lbann.Model = modeling.create_masked_language_modeling_transformer(
        dataset,
        embed_dim=transformer.hidden_size,
        num_encoders=0,
        num_decoders=len(transformer.layers),
        num_heads=transformer.num_heads,
        dropout=args.dropout,
        input_dropout=args.dropout,
        num_epochs=args.num_epochs,
        args=args,
        transformer=transformer,
    )
    lbann.util.amp.enable_amp(model, args)

    # Training schedule
    lr_decay_ratio = 1.0
    warmup_ratio = 40000 / 500000  # From paper
    sched_mult = args.dataset_fraction if args.fractional_schedule else 1.0
    total_steps = math.ceil(sched_mult * dataset.num_train_samples() /
                            args.mini_batch_size)

    train_script: BatchScript = trainer.construct_training_task(
        model,
        args,
        learning_rate=4e-4,
        # Training parameters from paper
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        clip_gradient=args.grad_clip,
        lr_decay='cosine',  # NOTE: The paper states linear decay
        lr_decay_steps=int(lr_decay_ratio * total_steps),
        end_learning_rate=4e-4 / 10,
        warmup_steps=int(warmup_ratio * total_steps),
        adamw_decay=0.01,
    )

    # Run trainer
    if not args.setup_only:
        train_script.run(overwrite=True)
    else:
        train_script.write(overwrite=True)


if __name__ == '__main__':
    main()
