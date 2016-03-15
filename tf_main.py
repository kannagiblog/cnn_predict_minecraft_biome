# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

from tensorflow.python.platform import gfile

from six.moves import xrange
import numpy as np
import tensorflow as tf

import tf_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("train_dir", "(train_dir)", "")
tf.app.flags.DEFINE_integer("max_steps", 30000, "")

def main(argv=None):
    if not gfile.Exists(FLAGS.train_dir):
        gfile.MakeDirs(FLAGS.train_dir)

    graph = tf.Graph()
    graph.device("/cpu:0")
    with graph.as_default():
        global_step = tf.Variable(0, trainable=False)
        images, labels = tf_model.inputs(training=True)
        logits = tf_model.inference(images)
        loss, accuracy = tf_model.loss(logits, labels)
        train_op = tf_model.train(loss, global_step)

        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.merge_all_summaries()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        with sess.as_default():
            if os.path.exists(FLAGS.train_dir + "checkpoint"):
                ckpt = tf.train.latest_checkpoint(FLAGS.train_dir)
                saver.restore(sess, ckpt)
            else:
                tf.initialize_all_variables().run()

            tf.train.start_queue_runners()

            summary_writer = tf.train.SummaryWriter(FLAGS.train_dir)

            for step in xrange(global_step.eval()+1, FLAGS.max_steps):
                start_time = time.time()
                _, loss_value, accuracy_value = sess.run([train_op, loss, accuracy])
                duration = time.time() - start_time

                if step % 5 == 0:
                    examples_per_sec = FLAGS.batch_size / duration
                    sec_par_batch = float(duration)
                    format_str = ("%s: step %d, loss = %.2f, accuracy = %.2f (%.1f examples/sec; %.3f sec/batch)")
                    print (format_str % (datetime.now(), step, loss_value, accuracy_value, examples_per_sec, sec_par_batch))

                    summary_str = summary_op.eval()
                    summary_writer.add_summary(summary_str, step)

                if step % 50 == 0 or (step+1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
                    saver.save(sess, checkpoint_path, global_step=step)

        sess.close()

if __name__ == '__main__':
    tf.app.run()
