#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
from https://stackoverflow.com/questions/50229091/is-it-necessary-to-close-session-after-tensorflow-interactivesession

Run in Spyder IDE
"""
import argparse
import tensorflow as tf
import numpy as np

from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesLimit
from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import MaxBytesInUse


def open_interactive_session():
    A = tf.Variable(np.random.randn(16, 255, 255, 3).astype(np.float32))
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())


def open_and_close_interactive_session():
    A = tf.Variable(np.random.randn(16, 255, 255, 3).astype(np.float32))
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.close()


def open_and_close_session():
    A = tf.Variable(np.random.randn(16, 255, 255, 3).astype(np.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

memstats = BytesInUse()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', help='repeat', type=int, default=5)
    parser.add_argument('type', choices=['interactive', 'interactive_close', 'normal'])
    args = parser.parse_args()

    sess_func = open_and_close_session


    if args.type == 'interactive':
        sess_func = open_interactive_session
    elif args.type == 'interactive_close':
        sess_func = open_and_close_interactive_session

    for _ in range(args.num):
        sess_func()
    with tf.Session() as sess:
#       print("bytes used=", sess.run(tf.contrib.memory_stats.BytesInUse()))
        print("bytes used=", sess.run(memstats))

#gives

"""
python example_session2.py interactive
('bytes used=', 405776640)
python example_session2.py interactive_close
('bytes used=', 7680)
python example_session2.py
('bytes used=', 7680)
"""
