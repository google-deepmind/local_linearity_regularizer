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
"""Batch normalisation."""

import sonnet as snt
import tensorflow.compat.v1 as tf


class BatchNorm(snt.AbstractModule):
  """Batch Normalisation layer that computes moving averages of statistics."""

  def __init__(self, decay_rate=0.1,
               scale=True, offset=True, eps=1e-5, name='BatchNorm'):
    super(BatchNorm, self).__init__(name=name)
    self.decay_rate = decay_rate
    self.eps = eps
    self.name = name
    self.use_scale = scale
    self.use_offset = offset

  def _build(
      self, inputs, is_training=True, test_local_stats=False,
      get_stats=None, set_stats=None):
    """Applies batch normalisation to the inputs.

    Args:
      inputs: Tensor to which batch normalisation is to be applied.
      is_training: Whether to update the exponential moving averages with
        the input's batch statistics.
      test_local_stats: If `True`, normalises the inputs using its own batch
        statistics. If `False`, normalises using the current exponential
        moving averages.
      get_stats: Optional `dict` to populate with batch stats.
      set_stats: Optional `dict` containing override values of batch stats.

    Returns:
      Normalised copy of `inputs`.
    """
    # Compute the batch statistics.
    axis = list(range(inputs.shape.ndims - 1))
    mean = tf.reduce_mean(inputs, axis=axis)
    mean_square = tf.reduce_mean(inputs ** 2, axis=axis)
    var = mean_square - mean ** 2

    if self.use_scale:
      self.gamma = tf.get_variable(
          'gamma',
          shape=mean.shape.as_list(),
          initializer=tf.ones_initializer(),
          trainable=True)
    else:
      self.gamma = None
    if self.use_offset:
      self.beta = tf.get_variable(
          'beta',
          shape=mean.shape.as_list(),
          initializer=tf.zeros_initializer(),
          trainable=True)
    else:
      self.beta = None

    # Exponential moving averages.
    accum_counter = tf.get_variable(
        'accumulation_counter',
        shape=[], dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=False)
    accum_mean = tf.get_variable(
        'accumulated_mean',
        shape=mean.shape.as_list(),
        initializer=tf.zeros_initializer(),
        trainable=False)
    accum_var = tf.get_variable(
        'accumulated_var',
        shape=mean.shape.as_list(),
        initializer=tf.ones_initializer(),
        trainable=False)

    # Handle getting/setting batch statistics.
    if get_stats is not None:
      # Get the batch statistics.
      # Save them into the `get_stats` dict. Use the `accum_xxx`
      # variables as keys, because we can rely on them being reused.
      get_stats[accum_mean] = mean
      get_stats[accum_var] = var
    elif set_stats is not None:
      # Set the batch statistics.
      # These are provided in the `set_stats` dict, populated by having
      # been passed to an earlier connection as `get_stats`.
      mean = set_stats[accum_mean]
      var = set_stats[accum_var]

    # Update the moving averages with the batch statistics, if in training mode.
    if is_training:
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.group([
          tf.assign_add(accum_counter, 1.),
          tf.assign_sub(accum_mean, self.decay_rate * (accum_mean - mean)),
          tf.assign_sub(accum_var, self.decay_rate * (accum_var - var)),
      ]))

    # Base the batch normalisation either on the moving averages,
    # or on the batch statistics.
    return tf.nn.batch_normalization(
        inputs,
        mean if test_local_stats else accum_mean,
        var if test_local_stats else accum_var,
        offset=self.beta, scale=self.gamma, variance_epsilon=self.eps)
