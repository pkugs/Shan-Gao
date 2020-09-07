import tensorflow as tf 
import numpy as np 
import tensorflow.contrib.layers as tcl
import time
from speech_data import data_reader
import config
import matplotlib.pyplot as plt
import os
import logging
class MetricChecker(object):
    def __init__(self, cfg, less=True):
        self.early_stop_count = cfg.early_stop_count
        self.reset_step()
        self.cur_dev = tf.placeholder(tf.float32, shape=[], name='cur_dev')
        if not less:
            self.best_dev = tf.get_variable(name='best_dev', trainable=False, shape=[],
                                            initializer=tf.constant_initializer(-np.inf))
            self.dev_improved = tf.less(self.best_dev, self.cur_dev)
        else:
            self.best_dev = tf.get_variable(name='best_dev', trainable=False, shape=[],
                                            initializer=tf.constant_initializer(np.inf))
            self.dev_improved = tf.less(self.cur_dev, self.best_dev)
        with tf.control_dependencies([self.dev_improved]):
            if not less:
                self.update_best_dev = tf.assign(self.best_dev,
                                                 tf.maximum(self.cur_dev, self.best_dev))
            else:
                self.update_best_dev = tf.assign(self.best_dev,
                                                 tf.minimum(self.cur_dev, self.best_dev))

    def reset_step(self):
        self.stop_step = 0

    def update(self, sess, cur_dev):
        dev_improved, best_dev = sess.run([self.dev_improved, self.update_best_dev],
                                          feed_dict={self.cur_dev: cur_dev})
        if dev_improved:
            self.reset_step()
        else:
            self.stop_step += 1
        return dev_improved, best_dev

    def should_stop(self):
        return self.stop_step >= self.early_stop_count

    def get_best(self, sess):
        return sess.run(self.best_dev)