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
"""Regularizers for robust training."""

from absl import logging
from interval_bound_propagation.src import attacks
import tensorflow.compat.v1 as tf

REGULARIZED_VARIABLES = 'REGULARIZED_VARIABLES'


def l2_regularization_loss(weight):
  """Returns L2 regularization loss on the trainable variables.

  Args:
    weight: Controls the size of the regularization loss. May be zero if no
      regularization is required.

  Returns:
    L2 regularization loss.
  """
  if not (tf.trainable_variables() and weight > 0):
    return tf.constant(0.)

  # Names of all variables that have already been regularized.
  regularization_losses = tf.get_collection(REGULARIZED_VARIABLES)
  reg_vars_names = {v.op.name for v in regularization_losses}

  def should_be_regularized(var):
    """Checks if a variable is a valid one to be regularized.

    A variable is valid if its name ends with '/w', 'w_dw' or '/w_pw' i.e.
    is not a bias or a batch norm variable. This function makes sure that
    a variable is not regularized twice.

    Args:
      var: A candidate tensorflow variable.

    Returns:
      A boolean to indicate whether the variable should be regularized.
    """
    var_name = var.op.name
    valid_name = var_name.endswith(('/w', '/w_dw', '/w_pw'))
    already_regularized = var_name in reg_vars_names
    return valid_name and not already_regularized

  def l2(tensor):
    with tf.name_scope(None, 'L2Regularizer', [tensor]):
      l2_weight = tf.convert_to_tensor(
          weight, dtype=tensor.dtype.base_dtype, name='weight')
      return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')

  regularized_variables = []
  reg_losses = []
  for var in tf.trainable_variables():
    if should_be_regularized(var):
      reg_losses.append(l2(var))
      tf.add_to_collection(REGULARIZED_VARIABLES, var)
      regularized_variables.append(var.op.name)
  logging.info('regularizing: %s', ', '.join(sorted(regularized_variables)))
  return tf.add_n(reg_losses, name='regularization_loss')


def local_linearity(
    model, inputs, labels, epsilon=8./255., num_steps=10, smoothing_factor=2.):
  """Finds the greatest violation of the linear surface in the epsilon-vicinity.

  Args:
    model: Callable taking (inputs, is_training, test_local_stats,
      get_stats, set_stats) and returning logits.
    inputs: Tensor containing nominal inputs to the model.
    labels: 1D int64 Tensor containing ground truth labels of the inputs.
    epsilon: Perturbation radius of the linearity attack.
    num_steps: Number of steps of PGD optimization.
    smoothing_factor: Weight placed on |d^T grad L|.

  Returns:
    0D Tensor containing the linearity attack loss.
  """
  batch_stats = {}
  logits = model(
      inputs, is_training=False, test_local_stats=True,
      get_stats=batch_stats)
  # Use one-hot probs instead of sparse labels.
  # This is necessary because tf.nn.sparse_softmax_cross_entropy_with_logits
  # does not allow taking the gradient of its gradient.
  probs = tf.one_hot(labels, logits.shape[-1].value)
  loss = tf.nn.softmax_cross_entropy_with_logits_v2(
      logits=logits, labels=probs)

  loss_grad = tf.gradients(loss, [inputs])[0]
  loss_grad_sg = tf.stop_gradient(loss_grad)

  def loss_fn(new_inputs, stop_grad=True, add_extra=False):
    """Local linearity measure for inputs."""
    new_logits = model(
        new_inputs, is_training=False, test_local_stats=True,
        set_stats=batch_stats)
    new_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=new_logits, labels=probs)

    perturbation = tf.stop_gradient(new_inputs - inputs)
    delta_grad = tf.reduce_sum(
        perturbation * (loss_grad_sg if stop_grad else loss_grad),
        axis=list(range(1, inputs.shape.ndims)))
    diff = new_loss - (loss + delta_grad)
    attack_loss = tf.sign(diff) * diff

    if add_extra:
      attack_loss += smoothing_factor * tf.sign(delta_grad) * delta_grad
    return -tf.reduce_mean(attack_loss)

  adv_x = attacks.pgd_attack(
      loss_fn,
      inputs,
      epsilon=epsilon,
      num_steps=num_steps,
      image_bounds=(0., 1.),
      optimizer=attacks.UnrolledAdam(0.1))

  return -loss_fn(adv_x, stop_grad=False, add_extra=True)
