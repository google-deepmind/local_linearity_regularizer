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
"""Tests for the training graph building."""

import functools

from absl.testing import absltest
import ml_collections
import sonnet as snt
import tensorflow.compat.v1 as tf

from local_linearity_regularizer import graph


class GraphTest(absltest.TestCase):

  def test_dataset_input_fns(self):
    train_batch_size = 5
    test_batch_size = 7

    image_height = 11
    image_width = 13
    image_channels = 3

    dataset_config = ml_collections.config_dict.ConfigDict({
        'image_size': (image_height, image_width, image_channels),
        'crop_size': (image_height, image_width, image_channels),
        'train_subset': '_train',
        'test_subset': '_test',
    })

    # Keep track of calls to the dataset constructor.
    dataset_instances = []
    def dataset_ctor(*args, **kwargs):
      dataset_instance = MockDataset(
          (image_height, image_width, image_channels), *args, **kwargs)
      dataset_instances.append(dataset_instance)
      return dataset_instance

    train_input_fn, test_input_fn = graph.dataset_input_fns(
        dataset_config, dataset_ctor, train_batch_size, test_batch_size)

    # The dataset constructor should have been called twice: once for the
    # train subset shuffled, and once for the test subset unshuffled.
    self.assertLen(dataset_instances, 2)
    self.assertEqual(train_batch_size, dataset_instances[0].batch_size)
    self.assertEqual('_train', dataset_instances[0].subset)
    self.assertEqual(True, dataset_instances[0].shuffle)
    self.assertEqual(test_batch_size, dataset_instances[1].batch_size)
    self.assertEqual('_test', dataset_instances[1].subset)
    self.assertEqual(False, dataset_instances[1].shuffle)

    # Check that each dataset instance yields batches of the correct size.
    train_batch = train_input_fn()
    self.assertEqual(
        [train_batch_size, image_height, image_width, image_channels],
        train_batch.shape.as_list())
    test_batch = test_input_fn()
    self.assertEqual(
        [test_batch_size, image_height, image_width, image_channels],
        test_batch.shape.as_list())

  def test_build_graph(self):
    train_batch_size = 5
    test_batch_size = 7

    config = ml_collections.config_dict.ConfigDict({
        'optimizer': self._optimizer_config(),
        'train': {
            'nominal_loss_weight': 1.,
            'adversarial_loss_weight': 0.,
            'l2_regularizer_weight': 0.,
            'llr': {
                'loss_weight': 0.,
            },
            'epsilon': .1,
            'train_pgd_steps': 20,
            'test_pgd_steps': 50,
        },
    })

    model = MockModel(num_classes=11)

    loss, train_measures, test_measures, _ = graph.build_graph(
        config,
        functools.partial(self._make_input, train_batch_size),
        functools.partial(self._make_input, test_batch_size),
        model_preprocess_fn=(lambda x: x),
        model=model,
    )

    # The module should be connected several times - see below.
    self.assertLen(model.connection_kwargs, 4)
    # Nominal training.
    self.assertEqual(True, model.connection_kwargs[0]['is_training'])
    self.assertEqual(True, model.connection_kwargs[0]['test_local_stats'])
    # Nominal testing.
    self.assertEqual(False, model.connection_kwargs[1]['is_training'])
    self.assertEqual(False, model.connection_kwargs[1]['test_local_stats'])
    # Adversarial testing: attack loop.
    self.assertEqual(False, model.connection_kwargs[2]['is_training'])
    self.assertEqual(False, model.connection_kwargs[2]['test_local_stats'])
    # Adversarial testing: evaluating adversarial example.
    self.assertEqual(False, model.connection_kwargs[3]['is_training'])
    self.assertEqual(False, model.connection_kwargs[3]['test_local_stats'])

    self._assert_is_scalar(loss)
    self._assert_is_scalar(train_measures['acc'])
    self._assert_is_scalar(test_measures['acc'])

  def test_model_preprocess(self):
    train_batch_size = 5
    test_batch_size = 7

    config = ml_collections.config_dict.ConfigDict({
        'optimizer': self._optimizer_config(),
        'train': {
            'nominal_loss_weight': 1.,
            'adversarial_loss_weight': 0.,
            'l2_regularizer_weight': 0.,
            'llr': {
                'loss_weight': 0.,
            },
            'epsilon': .1,
            'train_pgd_steps': 20,
            'test_pgd_steps': 50,
        },
    })

    # Keep track of invocations of the preprocess function.
    model_preprocess_calls = []
    def model_preprocess_fn(x):
      model_preprocess_calls.append(x)
      return 1. - x

    graph.build_graph(
        config,
        functools.partial(self._make_input, train_batch_size),
        functools.partial(self._make_input, test_batch_size),
        model_preprocess_fn=model_preprocess_fn,
        model=MockModel(num_classes=11),
    )

    # Check that the preprocess function was called for train and test inputs.
    self.assertLen(model_preprocess_calls, 4)
    # Nominal training.
    self.assertEqual(train_batch_size, model_preprocess_calls[0].shape[0].value)
    # Nominal testing.
    self.assertEqual(test_batch_size, model_preprocess_calls[1].shape[0].value)
    # Adversarial testing: attack loop.
    self.assertEqual(test_batch_size, model_preprocess_calls[2].shape[0].value)
    # Adversarial testing: evaluating adversarial example.
    self.assertEqual(test_batch_size, model_preprocess_calls[3].shape[0].value)

  def _optimizer_config(self):
    return {
        'learning_rate': 1.e-2,
        'lr_boundaries': [1000],
        'lr_decay_factor': .5,
        'momentum': .9,
    }

  def _make_input(
      self, batch_size, image_height=17, image_width=19, image_channels=3):

    return {
        'image': tf.placeholder(dtype=tf.float32, shape=[
            batch_size, image_height, image_width, image_channels]),
        'label': tf.placeholder(dtype=tf.int64, shape=[batch_size]),
    }

  def _assert_is_scalar(self, x):
    self.assertEqual(x.dtype, tf.float32)
    self.assertEqual(x.shape.as_list(), [])


class MockDataset(object):
  """Trivial dataset that records its call arguments."""

  def __init__(
      self, image_shape, batch_size, subset, shuffle=True, preprocess_fn=None):
    super().__init__()
    self._image_shape = image_shape
    self._batch_size = batch_size
    self._subset = subset
    self._shuffle = shuffle
    self._preprocess_fn = preprocess_fn

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def subset(self):
    return self._subset

  @property
  def shuffle(self):
    return self._shuffle

  def __call__(self):
    return self._make_one_shot_iterator().get_next()

  def _make_one_shot_iterator(self):
    return self

  def get_next(self):
    return tf.placeholder(
        dtype=tf.float32, shape=[self._batch_size, *self._image_shape])


class MockModel(snt.AbstractModule):
  """Trivial Sonnet module that records its call arguments."""

  def __init__(self, num_classes):
    super().__init__()
    self._num_classes = num_classes
    self._connection_kwargs = []

  @property
  def connection_kwargs(self):
    return self._connection_kwargs

  def _build(self, inputs, **kwargs):
    # Log the arguments for this connection to the graph.
    self._connection_kwargs.append(kwargs)

    mean = tf.reduce_mean(inputs, axis=list(range(1, inputs.shape.ndims)))
    b = tf.get_variable('b', dtype=tf.float32, shape=[self._num_classes])
    return tf.expand_dims(mean, 1) + b


if __name__ == '__main__':
  absltest.main()
