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
"""Training script using Wide ResNet.

To run locally:
python3 train_main.py --config=config.py
"""
import functools

from absl import app
from absl import flags
from ml_collections import config_flags
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from local_linearity_regularizer import datasets
from local_linearity_regularizer import graph
from local_linearity_regularizer import training_loop
from local_linearity_regularizer import wide_resnet


flags.DEFINE_integer('train_batch_size', 64, 'Batch size for training.')
flags.DEFINE_integer('num_runs', 100000, 'Number of training steps to execute.')
flags.DEFINE_integer('test_every', 4000,
                     'Number of runs between accuracy tests (<=0 to disable).')
flags.DEFINE_integer('test_batch_size', 80,
                     'Batch size to use for accuracy tests.')
flags.DEFINE_integer('num_test_runs', 125,
                     'Number of test set batches for accuracy tests.')
config_flags.DEFINE_config_file(
    'config', 'config.py', 'Configuration file for the experimental setup.')
FLAGS = flags.FLAGS


class NullCheckpoint(training_loop.Checkpoint):
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


def tfds_preprocess(batch):
  """Map image pixel values from uint8 [0,255] to float32 [0,1]."""
  return {
      'image': tf.cast(batch['image'], tf.float32) / 255.,
      'label': batch['label'],
  }


def tfds_load(name, batch_size, subset, shuffle, preprocess_fn=None):
  """Loads a TFDS dataset, with shuffle/preprocess dictated by the training.

  Args:
    name: TFDS name, e.g. 'cifar10'.
    batch_size: Batch size.
    subset: Dataset split, e.g. 'train'.
    shuffle: Whether to shuffle the dataset.
    preprocess_fn: Preprocessing requested by the training algorithm, for
      example random cropping.

  Returns:
    TF Dataset structured as {'image': batch of HWC images with values in [0,1],
      'label': batch of integer labels}.
  """
  ds = tfds.load(name, split=subset).cache().repeat()
  if shuffle:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  ds = ds.map(tfds_preprocess)
  if preprocess_fn:
    ds = ds.map(preprocess_fn)
  return ds


def main(_):
  config = FLAGS.config

  # Train and test datasets.
  if config.dataset == 'cifar10':
    model_preprocess_fn = datasets.cifar10_model_preprocess

  elif config.dataset == 'mnist':
    model_preprocess_fn = datasets.mnist_model_preprocess

  else:
    raise ValueError('Unsupported dataset: ' + config.dataset)

  _, dataset_info = tfds.load(config.dataset, split='train', with_info=True)
  num_classes = dataset_info.features['label'].num_classes

  train_input_fn, test_input_fn = graph.dataset_input_fns(
      config[config.dataset],
      functools.partial(tfds_load, config.dataset),
      FLAGS.train_batch_size,
      FLAGS.test_batch_size)

  # Classification model.
  model = wide_resnet.WideResNet(
      num_classes=num_classes, **config.model.kwargs.to_dict())

  loss, train_measures, test_measures, init_step_fn = graph.build_graph(
      config, train_input_fn, test_input_fn, model_preprocess_fn, model)

  checkpoint = NullCheckpoint()

  training_loop.train(
      loss, train_measures, test_measures, init_step_fn, checkpoint,
      num_runs=FLAGS.num_runs,
      test_every=FLAGS.test_every, num_test_runs=FLAGS.num_test_runs)


if __name__ == '__main__':
  tf.enable_resource_variables()
  app.run(main)
