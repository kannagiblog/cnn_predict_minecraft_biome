# coding: utf-8

from __future__ import absolute_import
from __future__ import division

import os

import tensorflow as tf
from tensorflow.python.platform import gfile

IMAGE_SIZE = 64

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes", 16, "")

def read(filename_queue):
    label_bytes = 1
    image_height = 128
    image_width = 128
    image_depth = 3
    image_bytes = image_height * image_width * image_depth
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]), [image_depth, image_height, image_width])
    image = tf.transpose(depth_major, [1, 2, 0])

    return image, label

def inputs(data_dir, batch_size, training=True):
    if training:
        filename = os.path.join(data_dir, "train_batch.bin")
        min_queue_examples = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filename = os.path.join(data_dir, "test_batch.bin")
        min_queue_examples = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    if not gfile.Exists(filename):
        raise ValueError("Failed to fin file: " + f)

    filename_queue = tf.train.string_input_producer([filename])

    read_image, read_label = read(filename_queue)
    read_image = tf.cast(read_image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    if training:
        read_image = tf.image.random_crop(read_image, [height, width])
        read_image = tf.image.random_flip_left_right(read_image)
        read_image = tf.image.random_flip_up_down(read_image)
        read_image = tf.image.random_brightness(read_image, max_delta=63)
        read_image = tf.image.random_contrast(read_image, lower=0.2, upper=1.8)
        read_image = tf.image.per_image_whitening(read_image)
    else:
        read_image = tf.image.resize_image_with_crop_or_pad(read_image, width, height)
        read_image = tf.image.per_image_whitening(read_image)

    images, labels = tf.train.shuffle_batch([read_image, read_label], batch_size=batch_size, num_threads=16, capacity=min_queue_examples + 3 * batch_size, min_after_dequeue=min_queue_examples)

    return images, tf.reshape(labels, [batch_size])
