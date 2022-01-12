# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""TensorFlow version of the Wide ResNet model from Zagoruyko et al 2016.

This implementation is modeled after the reference implementation at:
https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch

References:
"Wide Residual Networks". (Zagoruyko et al 2016).
https://arxiv.org/pdf/1605.07146.pdf
"Deep Residual Learning for Image Recognition" (He et al 2015a).
https://arxiv.org/pdf/1512.03385.pdf
"Identity Mappings in Deep Residual Networks" (He et al 2015b)
https://arxiv.org/pdf/1603.05027.pdf
"""

import sonnet as snt
import tensorflow.compat.v1 as tf

from local_linearity_regularizer import batchnorm


def he_normal_initializer(seed=None, dtype=tf.float32):
  """He initialization (also called MSR or MSRA initialization).

  Args:
    seed: A Python integer. Used to create random seeds.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer.

  Compared to Glorot initialization (TF default), He initialization uses double
  the standard deviation (becaue of ReLUs), and typically uses Gaussians rather
  than uniform distributions.

  ResNet papers typically use He initialization, and it tends to result slightly
  (~0.2%) higher accuracy, though sometimes with BatchNorm this doesn't matter
  at all.
  """
  return tf.variance_scaling_initializer(
      scale=2.0, mode='fan_in', distribution='normal', seed=seed, dtype=dtype)


_DEFAULT_INITIALIZERS = {'w': he_normal_initializer()}


def _bn_activations(
    inputs, activation='relu', activation_kwargs=None, decay_rate=0.1,
    **batchnorm_kwargs):
  """Batch norm plus activation layer.

  Args:
    inputs: 4D NHWC tensor of pre-activations.
    activation: Name of activation function within `tf.nn`.
    activation_kwargs: Additional keyword arguments to `activation`.
    decay_rate: Batch norm decay rate.
    **batchnorm_kwargs: Keyword arguments to pass through to the batch
      normalisation layer.

  Returns:
    4D NHWC tensor of activations.
  """
  activation_kwargs = activation_kwargs or {}
  batchnorm_layer = batchnorm.BatchNorm(
      decay_rate=decay_rate, scale=True, offset=True)
  pre_activations = batchnorm_layer(inputs, **batchnorm_kwargs)
  return getattr(tf.nn, activation)(pre_activations, **activation_kwargs)


class WideResNetBlock(snt.AbstractModule):
  """Wide ResNet Block, as implemented in Zagoruyko 2016.

  Specifically, this is a pre-activation ResNet block, described in He et al
  2015b, where each block consists of repeated applications of
  (BN - ReLU - Conv). See Fig. 2b from He et al 2015b for further explanation.

  All filters use a 3x3 receptive field. When `num_bottleneck_layers=1`, this
  corresponds to the "basic" block in Zagoruyko et al, which is the recommended
  block. This also corresponds to the original block structure from He et al
  2015a.

  The last difference is that when projection shortcuts are used, the
  convolutional shortcut layer is applied to the result of the activation,
  rather than to the original input. I haven't noticed this documented in the
  papers, but I suspect the idea is that there are never two consecutive
  convolutional layers without a batch norm layer in between.
  """

  def __init__(self,
               num_filters,
               num_bottleneck_layers=1,
               stride=1,
               projection_shortcut=False,
               activation='relu',
               activation_kwargs=None,
               decay_rate=0.1,
               initializers=None,
               name='wide_res_net_block'):
    """A Wide ResNet block, as in Zagoruyko 2016.

    Args:
      num_filters: An integer, the output dimension of each convolutional layer
        in the block.
      num_bottleneck_layers: An integer. The number of convolutional layers
        between residual connections is `num_bottleneck_layers + 1`. The Wide
        ResNet paper recommends that this be fixed at 1, and that depth instead
        by modified by changing the number of blocks per layer.
      stride: An integer. Stride greater than 1 will spatially downsample the
        input. Typically, this is 2 for the first block in every layer (except
        the first), and 1 for every other block.
      projection_shortcut: A boolean. If `False` (default), an identity
        connection is used. Otherwise a convolutional layer is applied after
        the first activation. This should be `True` whenever n_in != n_out or
        stride > 1. See discussion in class documentation above.
      activation: Name of activation function within `tf.nn`.
      activation_kwargs: Additional keyword arguments to `activation`.
      decay_rate: Batch norm decay rate.
      initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b'). The default initializers are
          the He or MSRA initializers. Use `initializers=None` for Glorot
          initialization (TF default).
      name: A string, the name of the Sonnet module.
    """
    super(WideResNetBlock, self).__init__(name=name)
    self._num_filters = num_filters
    self._num_bottleneck_layers = num_bottleneck_layers
    self._stride = stride
    self._projection_shortcut = projection_shortcut
    if stride > 1 and not projection_shortcut:
      raise ValueError('need a projection shortcut with stride > 1')
    self._initializers = initializers or _DEFAULT_INITIALIZERS
    self._activation = activation
    self._activation_kwargs = activation_kwargs
    self._decay_rate = decay_rate

  def _build(self, x, **batchnorm_kwargs):
    orig_x = x
    for i in range(self._num_bottleneck_layers + 1):
      stride = self._stride if i == 0 else 1
      x = _bn_activations(
          x,
          activation=self._activation,
          activation_kwargs=self._activation_kwargs,
          decay_rate=self._decay_rate,
          **batchnorm_kwargs)
      # When using a projection shortcut, the residual connection should
      # be applied from after the first activation, rather than before.
      if self._projection_shortcut and i == 0:
        orig_x = x
      x = snt.Conv2D(
          self._num_filters, [3, 3], stride=stride, use_bias=False,
          initializers=self._initializers, name='conv_{}'.format(i))(x)

    if self._projection_shortcut:
      shortcut_x = snt.Conv2D(
          self._num_filters, [1, 1], stride=self._stride, use_bias=False,
          initializers=self._initializers, name='shortcut_x')(orig_x)
      x += shortcut_x
    else:
      x += orig_x
    return x


class WideResNet(snt.AbstractModule):
  """Wide ResNet architecture used for experiments in Zagoruyko et al 2016.

  For best results, training and evaluation should use the CIFAR-10 pre-
  processing functions defined in `datasets.py`. This model achieves 96.0%
  accuracy on the CIFAR10 test set.

  A ResNet model consists of a single 3x3 convolutional layer, followed by three
  ResNet "layers", followed by a non-linearity, spatial average pooling, and a
  final linear layer.

  Each ResNet "layer" (not to be confused with a single layer of a neural
  network) consists of a number of ResNet Blocks, which is determined by the
  `depth` parameter.

  Within each layer, the first ResNet block should use a projection shortcut,
  and all others should use identity shortcuts. Each layer also operates at a
  different spatial resolution (32x32, 16x16, and 8x8), which means that the
  first ResNet block in each ResNet layer, other than the first, should also
  use a stride of 2.
  """

  def __init__(self,
               depth=28,
               width=10,
               num_classes=10,
               activation='relu',
               activation_kwargs=None,
               decay_rate=0.1,
               initializers=None,
               name='wide_res_net'):
    """A Wide ResNet block, as in Zagoruyko 2016.

    Args:
      depth: Number of layers. Must be 6n+4 for some positive integer n.
      width: Controls the number of channels per hidden layer. The number of
        channels will be initially `16*num_width`, increasing to `64*num_width`.
      num_classes: Number of output classes.
      activation: Name of activation function within `tf.nn`.
      activation_kwargs: Additional keyword arguments to `activation`.
      decay_rate: Batch norm decay rate.
      initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b'). The default initializers are
          the He or MSRA initializers. Use `initializers=None` for Glorot
          initialization (TF default).
      name: A string, the name of the Sonnet module.
    """
    super(WideResNet, self).__init__(name=name)
    if (depth - 4) % 6 != 0 or depth < 10:
      raise ValueError('depth is {}, but must be 6n+4'.format(depth))
    self._depth = depth
    self._width = width
    self._num_classes = num_classes
    self._initializers = initializers or _DEFAULT_INITIALIZERS
    self._activation = activation
    self._activation_kwargs = activation_kwargs
    self._decay_rate = decay_rate

  def _build(self, x, **batchnorm_kwargs):
    blocks_per_layer = (self._depth - 4) // 6
    filter_sizes = [self._width * n for n in [16, 32, 64]]

    x = snt.Conv2D(
        filter_sizes[0], [3, 3], stride=1, use_bias=False,
        initializers=self._initializers, name='init_conv')(x)
    for layer_num, filter_size in enumerate(filter_sizes):
      for i in range(blocks_per_layer):
        stride = 2 if (layer_num != 0 and i == 0) else 1
        projection_shortcut = (i == 0)
        block = WideResNetBlock(
            filter_size,
            num_bottleneck_layers=1,
            stride=stride,
            projection_shortcut=projection_shortcut,
            activation=self._activation,
            activation_kwargs=self._activation_kwargs,
            decay_rate=self._decay_rate,
            initializers=self._initializers,
            name='resnet_lay_{}_block_{}'.format(layer_num, i))
        x = block(x, **batchnorm_kwargs)
    x = _bn_activations(
        x,
        activation=self._activation,
        activation_kwargs=self._activation_kwargs,
        decay_rate=self._decay_rate,
        **batchnorm_kwargs)
    x = tf.reduce_mean(x, axis=[1, 2], keepdims=False, name='avg_pool')
    x = snt.Linear(self._num_classes, initializers=self._initializers)(x)
    return x


