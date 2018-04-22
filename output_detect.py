#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from io import StringIO
import os
import sys

import numpy as np
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import zipfile
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, sess):

    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in
                        op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[
                key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'],
                                     [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'],
                                     [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0],
                                     tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0],
                                   [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0],
                                   [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0],
            image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name(
        'image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={
                               image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(
        output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][
            0]
    return output_dict


def get_training_data_file_name_list():
    file_names = []
    with open('_home_lyk_machine_learning_Supervised_Learning_iNaturalist_image_train__7_dataset.txt') as f:
        file_label_pairs = f.readlines()
        for line in file_label_pairs:
            file = line.split('\t')[0]
            file_names.append(file)


def get_test_data_file_name_list():
    #return tf.gfile.Glob('/media/01/home/lyk/machine_learning/Supervised_Learning/iNaturalist_2018_Competition/raw_data/test2018/*.jpg')
    return tf.gfile.Glob('test2018/*.jpg')


NUM_CLASSES = 1
THRESHOLD = 0.9

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(
            'output_inference_graph/frozen_inference_graph.pb',
            'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(
    'inat_2017_label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# PATH_TO_TEST_IMAGES_DIR = 'image_data'
# TEST_IMAGE_PATHS = [
#     os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in
#     range(1, 3)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

"""
1.eliminate region size < (1/5 * orginal image size)
2.log image which has multiple regions and iou equal to 0 between those regions
"""

with detection_graph.as_default():
    with tf.Session() as sess:

        #for image_path in TEST_IMAGE_PATHS:
        file_name_list = get_test_data_file_name_list()
        file_name = file_name_list[0]

        path, _ = os.path.split(file_name)
        path = "{0}_object_detected".format(path)
        if not os.path.exists(path):
            os.mkdir(path)

        for image_path in file_name_list:
            image = Image.open(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.

            filter_out_dict = {}
            filter_out_dict['detection_scores'] = []
            filter_out_dict['detection_classes'] = []
            filter_out_dict['detection_boxes'] = []
            output_dict = run_inference_for_single_image(image_np, sess)
            for i in range(len(output_dict['detection_scores'])):
                if output_dict['detection_classes'][i] != 1:
                    continue
                if output_dict['detection_scores'][i] < THRESHOLD:
                    continue

                ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i]
                if (ymax - ymin) * (xmax - xmin) < 0.03:
                    continue

                filter_out_dict['detection_classes'].append(output_dict['detection_classes'][i])
                filter_out_dict['detection_scores'].append(output_dict['detection_scores'][i])
                filter_out_dict['detection_boxes'].append([ymin, xmin, ymax, xmax])

            _, name = os.path.split(image_path)
            name = name.split('.')[0]
            dest_path = os.path.join(path, name)
            if not os.path.exists(dest_path):
                os.mkdir(dest_path)
            if len(filter_out_dict['detection_classes']) == 0:
                image.save(os.path.join(dest_path, 'orginal.jpg'))
            else:
                image_width, image_height = image.size
                for i in range(len(filter_out_dict['detection_classes'])):
                    ymin, xmin, ymax, xmax = filter_out_dict['detection_boxes'][i]
                    cropped = image.crop((int(xmin * image_width * 0.9), int(ymin * image_height * 0.9),
                                          int(xmax * image_width * 1.1), int(ymax * image_height * 1.1)))
                    cropped.save(os.path.join(dest_path, '{0}.jpg'.format(i)))


            # filter_out_dict['detection_boxes'] = np.array(filter_out_dict['detection_boxes'])
            # # Visualization of the results of a detection.
            # vis_util.visualize_boxes_and_labels_on_image_array(
            #     image_np,
            #     filter_out_dict['detection_boxes'],
            #     filter_out_dict['detection_classes'],
            #     filter_out_dict['detection_scores'],
            #     category_index,
            #     instance_masks=output_dict.get('detection_masks'),
            #     use_normalized_coordinates=True,
            #     line_thickness=4,
            #     min_score_thresh=THRESHOLD)
            #
            # print(image_path)
            # plt.figure(figsize=IMAGE_SIZE)
            # plt.imshow(image_np)
            # plt.waitforbuttonpress()
            # plt.close()
