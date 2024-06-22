"""Normalization modules for transformer models."""
import lbann
from ..base import Module
from lbann.util import make_iterable


class LayerNorm(Module):
    """See https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html"""

    global_count = 0  # Static counter, used for default names

    def __init__(self, normalized_shape, name=None, builtin=True, eps=None):
        super().__init__()
        LayerNorm.global_count += 1
        self.normalized_shape = make_iterable(normalized_shape)
        self.name = (name if name else f'layernorm{LayerNorm.global_count}')
        self.builtin = builtin
        self.epsilon = eps

        # Initialize weights
        self.weight = lbann.Weights(
            initializer=lbann.ConstantInitializer(value=1),
            name=f'{self.name}_weight',
        )
        self.bias = lbann.Weights(
            initializer=lbann.ConstantInitializer(value=0),
            name=f'{self.name}_bias',
        )

    def forward(self, x, **extra_kwargs):
        if self.builtin:
            return lbann.LayerNorm(x,
                                   scale=True,
                                   bias=True,
                                   start_dim=-1,
                                   name=self.name,
                                   weights=[self.weight, self.bias],
                                   epsilon=self.epsilon,
                                   **extra_kwargs)

        # Normalization
        x = lbann.InstanceNorm(x, epsilon=self.epsilon, **extra_kwargs)

        # Affine transform
        s = lbann.WeightsLayer(
            weights=self.weight,
            dims=[1] + list(make_iterable(self.normalized_shape)),
            **extra_kwargs,
        )
        s = lbann.Tessellate(s, hint_layer=x, **extra_kwargs)
        b = lbann.WeightsLayer(
            weights=self.bias,
            dims=[1] + list(make_iterable(self.normalized_shape)),
            **extra_kwargs,
        )
        b = lbann.Tessellate(b, hint_layer=x, **extra_kwargs)
        x = lbann.Add(lbann.Multiply(s, x, **extra_kwargs), b, **extra_kwargs)
        return x
