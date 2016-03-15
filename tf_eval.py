# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

import tf_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("eval_dir", "(eval_dir)", "")
tf.app.flags.DEFINE_string("checkpoint_dir", "(checkpoint_dir)", "")
tf.app.flags.DEFINE_integer("num_iterate", 100, "")

def score(logits, labels, num_classes, sess):
    np_logits, np_labels = sess.run([logits, labels])

    accuracy = accuracy_score(np_labels, np_logits)
    precision = precision_score(np_labels, np_logits, average=None)
    recall = recall_score(np_labels, np_logits, average=None)
    confusion = confusion_matrix(np_labels, np_logits)

    return accuracy, precision, recall, confusion

def main(argv=None):
    if not gfile.Exists(FLAGS.eval_dir):
        gfile.MakeDirs(FLAGS.eval_dir)

    with tf.Graph().as_default():
        images, labels = tf_model.inputs(training=False)
        logits = tf_model.inference(images)
        logits = tf.squeeze(tf.argmax(logits, 1))

        saver = tf.train.Saver()

        graph_def = tf.get_default_graph().as_graph_def()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]

            coord = tf.train.Coordinator()
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            accuracy = 0
            precision = np.zeros(FLAGS.num_classes)
            recall = np.zeros(FLAGS.num_classes)
            confusion = np.zeros((FLAGS.num_classes, FLAGS.num_classes))
            for i in xrange(0, FLAGS.num_iterate):
                batch_accuracy, batch_precision, batch_recall, batch_confusion = score(logits, labels, num_classes=FLAGS.num_classes, sess=sess)
                accuracy += batch_accuracy
                precision += batch_precision
                recall += batch_recall
                confusion += batch_confusion

            print(accuracy / FLAGS.num_iterate)
            print(np.divide(precision, FLAGS.num_iterate))
            print(np.divide(recall, FLAGS.num_iterate))
            print(confusion)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

if __name__ == '__main__':
    tf.app.run()
