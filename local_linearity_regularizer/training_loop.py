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
"""Robust training and evaluation loop."""

import abc

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf


class Checkpoint(metaclass=abc.ABCMeta):
  """Interface for regularly saving progress, in case of interruption."""

  @abc.abstractmethod
  def restore_or_save(self, session):
    """Restores an existing checkpoint if any; otherwise saves an initial one.

    If a checkpoint is already present from an earlier abortive run (as
    determined by the implementation), it is loaded so that training can
    resume from where it left off. Both the TensorFlow trainable variables and
    the Python loop state are restored.

    If no such checkpoint is present, so there have been no earlier partial
    attempts at this training, an initial checkpoint is saved.

    Args:
      session: TensorFlow session.
    """

  @abc.abstractmethod
  def save(self, session, step):
    """Saves a new checkpoint.

    Both the TensorFlow trainable variables and the Python loop state are saved,
    in case the run is accidentally interrupted.

    Args:
      session: TensorFlow session.
      step: Training step, so that the implementation can elect to save a
        checkpoint only every n steps.
    """

  @property
  @abc.abstractmethod
  def state(self):
    """String-keyed dictionary of Python loop state."""


def train(loss, train_measures, test_measures, init_step_fn, checkpoint,
          num_runs,
          test_every=0, num_test_runs=0,
          master='', train_writer=None, test_writer=None):
  """Builds and trains model.

  Args:
    loss: 0D tensor containing the loss to be minimised.
    train_measures: Dict (with string keys) of 0D tensors containing
      training measurements.
    test_measures: Dict (with string keys) of 0D tensors containing
      test set evaluation measurements.
    init_step_fn: Function taking (session, initial_step_val)
      to be invoked to initialise the global training step.
    checkpoint: Checkpoint to save progress for resilience against interruption.
    num_runs: Number of training runs to perform. Each run consists of
      several training steps, as specified by `num_steps_per_run`.
    test_every: Frequency (expressed as 'every n training runs') with which to
      perform a full test set evaluation. Zero means never.
    num_test_runs: Number of test runs to perform in order to complete a full
      test set evaluation.
    master: Name of the TensorFlow master to use.
    train_writer: Optional XData writer to record `train_measures`.
    test_writer: Optional XData writer to record `test_measures`.
  """
  checkpoint.state['completed_runs'] = 0

  full_test_set_evaluation = _make_full_test_set_evaluation_fn(
      test_measures, num_test_runs, test_writer=test_writer)

  with tf.train.MonitoredTrainingSession(
      master=master, is_chief=True, chief_only_hooks=[], hooks=[],
      config=tf.ConfigProto(
          allow_soft_placement=True)
      ) as session:

    checkpoint.restore_or_save(session._tf_sess())  # pylint: disable=protected-access

    # Initialise the global training step.
    # This is necessary because it's not stored in the checkpoint.
    initial_step_val = checkpoint.state['completed_runs']
    init_step_fn(session, initial_step_val)

    while checkpoint.state['completed_runs'] < num_runs:
      # Training step.
      loss_val, train_measure_vals = session.run([loss, train_measures])
      run = checkpoint.state['completed_runs'] + 1
      step = run

      logging.info(
          '[%d] loss = %f, training measurements: %s',
          step, loss_val, train_measure_vals)
      if train_writer is not None:
        train_writer.write({
            'step': step,
            **{'train_' + key: val for key, val in train_measure_vals.items()},
        })

      if test_every > 0 and run % test_every == 0:
        full_test_set_evaluation(session, step)

      checkpoint.state['completed_runs'] = run
      checkpoint.save(session._tf_sess(), run)  # pylint: disable=protected-access


def evaluate(test_measures, num_test_runs=0, master='', test_writer=None):
  """Builds and trains model.

  Args:
    test_measures: Dict (with string keys) of 0D tensors containing
      test set evaluation measurements.
    num_test_runs: Number of test runs to perform in order to complete a full
      test set evaluation.
    master: Name of the TensorFlow master to use.
    test_writer: Optional XData writer to record `test_measures`.
  """
  full_test_set_evaluation = _make_full_test_set_evaluation_fn(
      test_measures, num_test_runs, test_writer=test_writer)

  with tf.train.MonitoredTrainingSession(
      master=master, is_chief=True, chief_only_hooks=[],
      hooks=[],
      config=tf.ConfigProto(
          allow_soft_placement=True)
      ) as session:
    # Use step=0 because there is only a single evaluation run in this case.
    full_test_set_evaluation(session, step=0)


def _make_full_test_set_evaluation_fn(
    test_measures, num_test_runs, test_writer=None):
  """Returns a function to perform a full test set evaluation.

  Args:
    test_measures: Dict (with string keys) of 0D tensors containing
      test set evaluation measurements.
    num_test_runs: Number of test runs to perform in order to complete a full
      test set evaluation.
    test_writer: Optional XData writer to record `test_measures`.

  Returns:
    Callable accepting `(session, step)` to perform a full test set evaluation.
  """
  def run(session, step):
    test_measure_vals = tf.nest.map_structure(
        lambda *x: np.mean(x),
        *[session.run(test_measures) for _ in range(num_test_runs)])

    logging.info(
        '[%d] test set evaluation on %d batches: %s',
        step, num_test_runs, test_measure_vals)
    if test_writer is not None:
      test_writer.write({
          'step': step,
          **{'test_' + key: val for key, val in test_measure_vals.items()},
      })

  return run

