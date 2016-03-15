# coding: utf-8

from __future__ import absolute_import
from __future__ import division

import re

import tensorflow.python.platform
import tensorflow as tf

import tf_input

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("batch_size", 128, "")
tf.app.flags.DEFINE_string("data_dir", "(data_dir)", "")

IMAGE_SIZE = tf_input.IMAGE_SIZE
NUM_CLASSES = FLAGS.num_classes
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = tf_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = tf_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

#MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 64.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.05

TOWER_NAME = "tower"

def _activation_summary(tensor):
    tensor_name = re.sub("%s_[0-9]*/" % TOWER_NAME, "", tensor.op.name)
    tf.histogram_summary(tensor_name + "/activations", tensor)
    tf.scalar_summary(tensor_name + "/sparsity", tf.nn.zero_fraction(tensor))

def _image_summary(tensor):
    # input -> [1, width, height, channels]
    with tf.variable_scope("imaging") as scope:
        images = tensor
        images = tf.transpose(images, [3, 1, 2, 0])
        tf.image_summary(tensor.op.name, images, max_images=64)

def _get_variable(name, shape, initializer, wd):
    var = tf.get_variable(name, shape, initializer=initializer)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name="weight_loss")
        tf.add_to_collection("losses", weight_decay)
    return var

def inputs(training):
    if not FLAGS.data_dir:
        raise ValueError("Please supply a data_dir")
    return tf_input.inputs(data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size, training=training)

def inference(images):
    tf.image_summary("input_images", tf.slice(images, [0, 0, 0, 0], [1, 64, 64, 3]), max_images=1)

    # conv_1
    with tf.variable_scope("conv1") as scope:
        kernel = _get_variable("weights", [7, 7, 3, 64], tf.truncated_normal_initializer(stddev=1e-4), 0.0)
        conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding="SAME")
        bias = _get_variable("biases", [64], tf.constant_initializer(0.0), 0.0)
        conv1 = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope.name)
        _activation_summary(conv1)
        _image_summary(tf.slice(conv1, [0, 0, 0, 0], [1, 64, 64, 64]))

    # pool_1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")

    # norm_1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=1e-4, beta=0.75, name="norm1")

    # local_2
    with tf.variable_scope("local2") as scope:
        dim = 1
        for d in norm1.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(norm1, [FLAGS.batch_size, dim])
        weights = _get_variable("weights", [dim, 256], tf.truncated_normal_initializer(stddev=0.04), 0.004)
        biases = _get_variable("biases", [256], tf.constant_initializer(0.1), 0.0)
        local2 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local2)

    # local_3
    with tf.variable_scope("local3") as scope:
        weights = _get_variable("weights", [256, 128], tf.truncated_normal_initializer(stddev=1/256), 0.004)
        biases = _get_variable("biases", [128], tf.constant_initializer(0.1), 0.0)
        local3 = tf.nn.relu(tf.matmul(local2, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # softmax
    with tf.variable_scope("softmax_linear") as scope:
        weights = _get_variable("weights", [128, NUM_CLASSES], tf.truncated_normal_initializer(stddev=1/128), 0.0)
        biases = _get_variable("biases", [NUM_CLASSES], tf.constant_initializer(0.0), 0.0)
        softmax_linear = tf.add(tf.matmul(local3, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear

def loss(logits, labels):
    sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
    indices = tf.reshape(tf.range(0, FLAGS.batch_size), [FLAGS.batch_size, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    dense_labels = tf.sparse_to_dense(concated, [FLAGS.batch_size, NUM_CLASSES], 1.0, 0.0)

    # cross_entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, dense_labels, name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
    tf.add_to_collection("losses", cross_entropy_mean)

    # accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(dense_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return tf.add_n(tf.get_collection("losses"), name="total_loss"), accuracy

def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name="avg")
    losses = tf.get_collection("losses")
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + " (raw)", l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op

def train(total_loss, global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
    tf.scalar_summary("learning_rate", lr)

    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + "/gradients", grad)

    #variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    #variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name="train")

    return train_op
