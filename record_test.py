#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

record_iterator = tf.python_io.tf_record_iterator('../iNaturalist_2017_object_detection/coco_val.record')

for record in record_iterator:
  example = tf.train.Example()
  example.ParseFromString(record)
  #print example.features.feature['image/encoded'].bytes_list.value[0]
  f = open('test.jpg', 'wb')
  f.write(example.features.feature['image/encoded'].bytes_list.value[0])
  f.close()
  print(example.features.feature['image/filename'].bytes_list.value[0])
  print(example.features.feature['image/object/class/label'].int64_list.value)
  break