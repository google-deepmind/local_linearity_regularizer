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
"""Configuration for `train_main.py`."""

from ml_collections import config_dict


def get_cifar10_config():
  config = config_dict.ConfigDict()
  config.crop_size = (32, 32, 3)
  config.image_size = (32, 32, 3)
  config.train_subset = 'train'
  config.test_subset = 'test'
  return config


def get_mnist_config():
  config = config_dict.ConfigDict()
  config.crop_size = (28, 28, 1)
  config.image_size = (28, 28, 1)
  config.train_subset = 'train'
  config.test_subset = 'test'
  return config


def get_wide_resnet_model_config():
  config = config_dict.ConfigDict()
  config.kwargs = config_dict.ConfigDict()
  config.kwargs.depth = 28
  config.kwargs.width = 8
  config.kwargs.activation = 'softplus'
  config.kwargs.activation_kwargs = {}
  return config


def get_optimizer_config():
  config = config_dict.ConfigDict()
  config.learning_rate = 0.1
  config.lr_decay_factor = 0.1
  config.lr_boundaries = (int(80e3), int(88e3), int(96e3))
  config.momentum = 0.9
  return config


def get_llr_config():
  """Local linearity regularisation."""
  config = config_dict.ConfigDict()
  config.loss_weight = 4.

  # Weight placed on |d^T grad L|.
  config.smoothing_factor = .3

  # PGD settings, for the search for point furthest from the linear approx.
  config.num_steps = 10
  config.epsilon = 8./255.

  config.warm_start = 500
  return config


def get_training_config():
  """Training configuration, including adversarial and regularisers."""
  config = config_dict.ConfigDict()
  config.l2_regularizer_weight = 0.0002
  config.nominal_loss_weight = 2.
  config.adversarial_loss_weight = 0.
  config.epsilon = 8./255.
  config.train_pgd_steps = 20
  config.test_pgd_steps = 50
  config.llr = get_llr_config()
  return config


def get_config():
  """Full experimental configuration for CIFAR."""
  config = config_dict.ConfigDict()
  config.dataset = 'cifar10'
  config.cifar10 = get_cifar10_config()
  config.mnist = get_mnist_config()
  config.model = get_wide_resnet_model_config()
  config.optimizer = get_optimizer_config()
  config.train = get_training_config()
  return config
