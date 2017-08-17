# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

import tensorflow.contrib.slim as slim

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, num_classes, activation_fn=tf.nn.softmax,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MyModel(models.BaseModel):
  """Logistic model with L2 regularization."""
  def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    model_input = tf.reshape(model_input,[-1,28,28,1])
    strides=1
    p_keep_conv=0.5
    p_keep_hidden=0.7
    W = tf.Variable(tf.random_normal([3, 3, 1, 32]))
    """w = tf.Variable(tf.random_normal([3, 3, 1, 32]))
    w2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))     
    w3 = tf.Variable(tf.random_normal([3, 3, 64, 128]))    
    w4 = tf.Variable(tf.random_normal([128 * 4 * 4, 625])) 
    w_o = tf.Variable(tf.random_normal([625, 10]))"""
    net=tf.nn.conv2d(model_input, W, strides=[1, strides, strides, 1], padding='SAME')
    net = slim.flatten(net)
    """l1a = tf.nn.relu(tf.nn.conv2d(model_input, w,                       
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],             
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)
    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)"""
    output = slim.fully_connected(
        net, num_classes, activation_fn=tf.nn.softmax,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}