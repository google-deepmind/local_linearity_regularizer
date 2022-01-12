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
"""Dataset preprocessing utilities."""

import functools

import numpy as np
import tensorflow.compat.v1 as tf


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]


def random_crop(image, image_shape, crop_shape):
  """Data augmentation used in training.

  Args:
    image: 4D tensor of shape (batch_size, image_height, image_width, channels)
    image_shape: A tuple of (image_height, image_width, channels)
    crop_shape: A tuple of (crop_height, crop_width, channels)

  Returns:
    fn: A cropping function, which takes a Tensor for a single image and
      returns a Tensor for a cropped version of the image.
  """
  has_batch = len(image.shape) > len(image_shape)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.reshape(image, [-1] + list(image_shape))
  image = tf.image.random_flip_left_right(image)
  image = tf.pad(
      image, paddings=[[0, 0], [4, 4], [4, 4], [0, 0]], mode='REFLECT')
  image = tf.random_crop(image, [tf.shape(image)[0]] + list(crop_shape))
  # Remove the batch dimension if it didn't originally have one.
  if not has_batch:
    image = tf.squeeze(image, axis=0)
  return image


def random_crop_preprocess_fn(image_shape, crop_shape):
  """Pre-processing to occur as part of the training dataset.

  Pre-processing for the training set entails random cropping. The test set
  receives no pre-processing: it's just the identity function.

  Args:
    image_shape: A tuple of (image_height, image_width, channels)
    crop_shape: A tuple of (crop_height, crop_width, channels)

  Returns:
    Data batch preprocessing function, accepting and returning a `dict` with
      keys 'image' and 'label'.
  """
  image_fn = functools.partial(
      random_crop, image_shape=image_shape, crop_shape=crop_shape)

  return lambda x: {'image': image_fn(x['image']), 'label': x['label']}


def cifar10_model_preprocess(image):
  """Processing which should be combined with model for adv eval.

  Performs an affine transform on each element of the image tensor to map them
  to a range that straddles zero. A slightly different transform is applied
  to each channel to reflect the statistics of the Cifar-10 dataset.

  Args:
    image: 4D NHWC image tensor with float32 values in the range [0, 1].

  Returns:
    4D NHWC image tensor with float32 values in the approximate range [-1, +1].
  """
  cifar_means = [125.3, 123.0, 113.9]
  cifar_devs = [63.0, 62.1, 66.7]
  rescaled_means = np.array([x / 255. for x in cifar_means])
  rescaled_devs = np.array([x / 255. for x in cifar_devs])
  return (image - rescaled_means) / rescaled_devs


def mnist_model_preprocess(image):
  """Processing which should be combined with model for adv eval."""
  return 2. * image - 1.

