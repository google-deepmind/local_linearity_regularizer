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
"""Builds the robust training graph."""

import collections
import functools

from interval_bound_propagation.src import attacks
import tensorflow.compat.v1 as tf

from local_linearity_regularizer import datasets
from local_linearity_regularizer import regularizers


DatasetInputFns = collections.namedtuple('DatasetInputFns', ['train', 'test'])


def dataset_input_fns(config, dataset_ctor, train_batch_size, test_batch_size):
  """Returns functions to create TensorFlow graph for loading from dataset.

  Args:
    config: Dataset configuration.
    dataset_ctor: Dataset's constructor.
    train_batch_size: Batch size for the training set.
    test_batch_size: Batch size for the test set.

  Returns:
    train_input_fn: Callable returning the training data as a nest of tensors.
    test_input_fn: Callable returning the test data as a nest of tensors.
  """
  train_preprocess_fn = datasets.random_crop_preprocess_fn(
      config.image_size, config.crop_size)

  train_dataset = dataset_ctor(
      train_batch_size,
      subset=config.train_subset,
      shuffle=True,
      preprocess_fn=train_preprocess_fn)
  train_input_fn = (
      lambda: tf.data.make_one_shot_iterator(train_dataset).get_next())

  test_dataset = dataset_ctor(
      test_batch_size,
      subset=config.test_subset,
      shuffle=False)
  test_input_fn = (
      lambda: tf.data.make_one_shot_iterator(test_dataset).get_next())

  return DatasetInputFns(train=train_input_fn, test=test_input_fn)


def _linear_schedule(step, init_step, final_step, init_value, final_value):
  """Returns a scalar interpolated between `init_value` and `final_value`."""
  rate = tf.cast(step - init_step, tf.float32) / float(final_step - init_step)
  linear_value = rate * (final_value - init_value) + init_value
  return tf.clip_by_value(
      linear_value, min(init_value, final_value), max(init_value, final_value))


def _top_k_accuracy(labels, logits, k=1):
  """Proportion of examples having true label amongst the top k predicted.

  Args:
    labels: 3D int64 tensor containing labels. Its shape is
      (num_steps_per_run, num_replicas, batch_size_per_replica).
    logits: 4D float32 tensor of shape (labels_shape..., num_classes)
      containing predicted logits.
    k: Positive integer.

  Returns:
    Float32 scalar containing proportion of examples whose true label is
      amongst the top `k` logits.
  """
  # Combine the leading dimensions into a single batch dimension.
  labels = tf.reshape(labels, [-1])
  logits = tf.reshape(logits, [-1, tf.shape(logits)[-1]])
  in_top_k = tf.nn.in_top_k(predictions=logits, targets=labels, k=k)
  return tf.reduce_mean(tf.cast(in_top_k, tf.float32))


def _optimizer(optimizer_config, global_step):
  """Returns the optimizer with the configured learning rate schedule."""
  lr_values = [optimizer_config.learning_rate]
  for _ in optimizer_config.lr_boundaries:
    lr_values.append(lr_values[-1] * optimizer_config.lr_decay_factor)
  learning_rate = tf.train.piecewise_constant(
      global_step,
      values=lr_values,
      boundaries=optimizer_config.lr_boundaries)
  return tf.train.MomentumOptimizer(learning_rate, optimizer_config.momentum)


def _margin_logit_loss(model_logits, true_class, clip_loss_value=-30.):
  """Computes difference between logit for true label and next highest."""
  num_classes = model_logits.get_shape().as_list()[-1]
  logit_mask = tf.one_hot(true_class, depth=num_classes, axis=-1)
  label_logits = tf.reduce_sum(logit_mask * model_logits, axis=-1)
  logits_with_true_label_neg_inf = model_logits - logit_mask * 10000
  highest_nonlabel_logits = tf.reduce_max(
      logits_with_true_label_neg_inf, axis=-1)
  loss = -(highest_nonlabel_logits - label_logits)
  loss = tf.maximum(loss, clip_loss_value)
  return tf.reduce_mean(loss)


def carlini_wagner_attack(
    model_fn, images, labels, image_bounds, epsilon,
    num_steps, learning_rate=0.1):
  """PGD using loss from Carlini-Wagner https://arxiv.org/abs/1608.04644.

  More specifically this implements:
  * Projected Gradient Descent [Madry, Kurakin]
  * on the margin loss described in [Carlini Wagner]
  * using the Adam optimizer [Kingma and Ba]

  These defaults are chosen as we find this attack to reliably converge to
  slightly better solutions than vanilla iterative FGSM.

  Note that while this method is named `carlini_wagner`, it does not implement
  the original attack described in the Carlini-Wagner paper. The CW attack
  treats the norm constraint as a Lagrangian penalty, and performs binary search
  on the penalty weight to find the smallest possible perturbation. Here, we
  use a fixed perturbation radius, and enforce this constraint with projection.

  Args:
    model_fn: Callable taking image tensor and returning logits tensor.
    images: Tensor of minibatch of images.
    labels: Integer 1D tensor containing true labels.
    image_bounds: Range of each element of `images`.
    epsilon: Admissible L-infinity perturbation radius for the attack.
    num_steps: Number of PGD iterations to perform.
    learning_rate: PGD learning rate.

  Returns:
   Tensor of adversarial images found by the attack.
  """
  def loss_fn(image):
    model_logits = model_fn(image)
    return _margin_logit_loss(model_logits, labels)

  return attacks.pgd_attack(
      loss_fn, images, epsilon, num_steps,
      optimizer=attacks.UnrolledAdam(learning_rate),
      image_bounds=image_bounds)


def _model_with_preprocess_fn(model, model_preprocess_fn):
  """Combines the model with its preprocessing function.

  Args:
    model: Callable taking (preprocessed_images, is_training, test_local_stats)
      and returning logits.
    model_preprocess_fn: Image pre-processing to be combined with `model`.

  Returns:
    Callable taking (raw_images, is_training, test_local_stats) and returning
    logits.
  """

  def model_with_preprocess(images, **batchnorm_kwargs):
    # Transform inputs from [0, 1] to roughly [-1, +1], which is the range used
    # by ResNets. (This may entail slightly different transformations for each
    # channel to reflect the dataset statistics.)
    # Do so here (instead of during dataset loading) because
    # adversarial attacks assume that the model inputs are [0, 1].
    images = model_preprocess_fn(images)
    return model(images, **batchnorm_kwargs)

  return model_with_preprocess


def _train_step(config, model, global_step, optimizer, batch):
  """Creates TensorFlow graph for a single training step.

  Args:
    config: Experiment configuration.
    model: Callable taking (images, is_training, test_local_stats) and
      returning logits.
    global_step: TensorFlow global step counter.
    optimizer: Training optimiser.
    batch: Dict of tensors for the minibatch, under keys 'image' and 'label'.

  Returns:
    loss: Training loss being minimised.
    logits: Nominal logits, of shape [batch_size, num_classes].
    adv_logits: Adversarial logits, of shape [batch_size, num_classes].
    labels: True labels of the examples in the minibatch.
  """
  images = batch['image']
  labels = batch['label']

  # Nominal.
  logits = model(images, is_training=True, test_local_stats=True)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels)

  # Adversarial.
  if config.adversarial_loss_weight > 0.:
    adv_images = carlini_wagner_attack(
        functools.partial(model, is_training=False, test_local_stats=False),
        images, labels,
        image_bounds=(0., 1.),
        epsilon=config.epsilon,
        num_steps=config.train_pgd_steps)
    adv_logits = model(adv_images, is_training=False, test_local_stats=True)
    adv_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=adv_logits, labels=labels)
  else:
    adv_logits = tf.zeros_like(logits)
    adv_cross_entropy = tf.zeros_like(cross_entropy)

  # Local linearity regulariser.
  if config.llr.loss_weight > 0.:
    linear_epsilon = _linear_schedule(
        global_step, 0, config.llr.warm_start, 0, config.llr.epsilon)
    linearity_loss = regularizers.local_linearity(
        model, images, labels,
        epsilon=linear_epsilon,
        num_steps=config.llr.num_steps,
        smoothing_factor=config.llr.smoothing_factor)
  else:
    linearity_loss = 0.

  # Combine the losses.
  classification_loss = tf.reduce_mean(
      config.nominal_loss_weight * cross_entropy +
      config.adversarial_loss_weight * adv_cross_entropy)
  regularization_loss = (
      regularizers.l2_regularization_loss(config.l2_regularizer_weight) +
      config.llr.loss_weight * linearity_loss)
  loss = tf.add(classification_loss, regularization_loss, name='loss')

  # Optimise.
  train_op = optimizer.minimize(loss, global_step=global_step)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops + [train_op]):
    loss = tf.identity(loss)

  return loss, logits, adv_logits, labels


def _test_step(config, model, batch):
  """Creates TensorFlow graph for a model evaluation step.

  Args:
    config: Experiment configuration.
    model: Callable taking (images, is_training, test_local_stats) and
      returning logits.
    batch: Dict of tensors for the minibatch, under keys 'image' and 'label'.

  Returns:
    logits: Nominal logits, of shape [batch_size, num_classes].
    adv_logits: Adversarial logits, of shape [batch_size, num_classes].
    labels: True labels of the examples in the minibatch.
  """
  images = batch['image']
  labels = batch['label']

  # Nominal.
  logits = model(images, is_training=False, test_local_stats=False)

  # Adversarial.
  adv_images = carlini_wagner_attack(
      functools.partial(model, is_training=False, test_local_stats=False),
      images, labels,
      image_bounds=(0., 1.),
      epsilon=config.epsilon,
      num_steps=config.test_pgd_steps)
  adv_logits = model(adv_images, is_training=False, test_local_stats=False)

  return logits, adv_logits, labels


def build_graph(
    config,
    train_input_fn, test_input_fn, model_preprocess_fn, model):
  """Builds the training graph.

  Args:
    config: Training configuration.
    train_input_fn: Callable returning the training data as a nest of tensors.
    test_input_fn: Callable returning the test data as a nest of tensors.
    model_preprocess_fn: Image pre-processing that should be combined with
      the model for adversarial evaluation.
    model: Callable taking (preprocessed_images, is_training, test_local_stats)
      and returning logits.

  Returns:
    loss: 0D tensor containing the loss to be minimised.
    train_measures: Dict (with string keys) of 0D tensors containing
      training measurements.
    test_measures: Dict (with string keys) of 0D tensors containing
      test set evaluation measurements.
    init_step_fn: Function taking (session, initial_step_val)
      to be invoked to initialise the global training step.
  """
  global_step = tf.train.get_or_create_global_step()
  optimizer = _optimizer(config.optimizer, global_step)

  model_with_preprocess = _model_with_preprocess_fn(
      model, model_preprocess_fn)

  # Training step.
  loss, train_logits, train_adv_logits, train_labels = _train_step(
      config.train, model_with_preprocess, global_step, optimizer,
      train_input_fn())
  train_measures = {
      'acc': _top_k_accuracy(train_labels, train_logits),
  }
  if config.train.adversarial_loss_weight > 0.:
    train_measures.update({
        'adv_acc': _top_k_accuracy(train_labels, train_adv_logits),
    })

  # Test evaluation.
  with tf.name_scope('test_accuracy'):
    test_logits, test_adv_logits, test_labels = _test_step(
        config.train, model_with_preprocess, test_input_fn())
    test_measures = {
        'acc': _top_k_accuracy(test_labels, test_logits),
        'adv_acc': _top_k_accuracy(test_labels, test_adv_logits),
    }

  initial_step = tf.placeholder(shape=(), dtype=tf.int64)
  init_global_step_op = tf.assign(global_step, initial_step)

  def init_step_fn(session, initial_step_val):
    session.run(init_global_step_op, feed_dict={initial_step: initial_step_val})

  return loss, train_measures, test_measures, init_step_fn


