"""Smoke tests for training the model, using mock data."""

import functools

from absl.testing import absltest
import ml_collections
import tensorflow.compat.v1 as tf

from local_linearity_regularizer import datasets
from local_linearity_regularizer import graph
from local_linearity_regularizer import training_loop
from local_linearity_regularizer import wide_resnet


class TrainTest(absltest.TestCase):

  def test_training(self):

    image_shape = (32, 32, 3)
    num_classes = 10
    train_batch_size = 5
    test_batch_size = 3

    # Train and test datasets.
    dataset_config = ml_collections.config_dict.ConfigDict({
        'image_size': image_shape,
        'crop_size': image_shape,
        'train_subset': 'train',
        'test_subset': 'test',
    })
    dataset_ctor = functools.partial(MockDataset, image_shape, num_classes)
    train_input_fn, test_input_fn = graph.dataset_input_fns(
        dataset_config, dataset_ctor, train_batch_size, test_batch_size)

    # Classification model.
    model = wide_resnet.WideResNet(
        num_classes=num_classes,
        depth=28, width=8, activation='softplus', activation_kwargs={})

    config = ml_collections.config_dict.ConfigDict({
        'optimizer': {
            'learning_rate': 1.e-2,
            'lr_boundaries': [4, 7],
            'lr_decay_factor': .5,
            'momentum': .9,
        },
        'train': {
            'nominal_loss_weight': 1.,
            'adversarial_loss_weight': 0.,
            'l2_regularizer_weight': 0.,
            'llr': {
                'loss_weight': .01,
                'smoothing_factor': .3,
                'num_steps': 2,
                'epsilon': .04,
                'warm_start': 2,
            },
            'epsilon': .03,
            'train_pgd_steps': 2,
            'test_pgd_steps': 3,
        },
    })

    loss, train_measures, test_measures, init_step_fn = graph.build_graph(
        config, train_input_fn, test_input_fn,
        datasets.cifar10_model_preprocess, model)

    checkpoint = MockCheckpoint()

    training_loop.train(
        loss, train_measures, test_measures, init_step_fn, checkpoint,
        num_runs=10,
        test_every=3, num_test_runs=2)


class MockCheckpoint(training_loop.Checkpoint):
  """Placeholder for checkpointing."""

  def __init__(self):
    super().__init__()
    self._state = {}

  def restore_or_save(self, session):
    pass

  def save(self, session, step):
    pass

  @property
  def state(self):
    return self._state


class MockDataset(object):
  """Dataset with random image pixels and labels."""

  def __init__(
      self, image_shape, num_classes, batch_size, *,
      subset, shuffle, preprocess_fn=(lambda x: x)):
    super().__init__()
    self._image_shape = image_shape
    self._num_classes = num_classes
    self._batch_size = batch_size
    self._subset = subset
    self._shuffle = shuffle
    self._preprocess_fn = preprocess_fn

  def __call__(self):
    return self._make_one_shot_iterator().get_next()

  def _make_one_shot_iterator(self):
    return self

  def get_next(self):
    data_batch = {
        'image': tf.random.uniform(
            minval=0., maxval=1.,
            dtype=tf.float32, shape=[self._batch_size, *self._image_shape]),
        'label': tf.random.uniform(
            maxval=self._num_classes,
            dtype=tf.int64, shape=[self._batch_size]),
    }
    if self._preprocess_fn is not None:
      data_batch = self._preprocess_fn(data_batch)
    return data_batch


if __name__ == '__main__':
  absltest.main()
