#!/usr/bin/env python
# coding: utf-8

import cv2
import sys

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import glob

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from IPython.display import display

# This is needed since the notebook is stored in the object_detection folder.

sys.path.append("..")
# sys.path.append("/home/tensorflow/models/research/")
# sys.path.append("/home/tensorflow/models/research/slim/")
# sys.path.append("/home/tensorflow/models/research/object_detection/")

from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# Path to frozen detection graph.
# This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './mscoco_label_map.pbtxt'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        cap = cv2.VideoCapture(1)
        cap.set(3, 320)  
        cap.set(4, 240)
        ret = 1
        i = 0
        while ret:

            ret, image_np = cap.read()
            if not ret:
                break
            image_np = cv2.resize(image_np, (320, 240))
            timer = cv2.getTickCount()

            # cv2.imshow("frame", image_np)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(q
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            image = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),  
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,
                min_score_thresh=0.8,
                line_thickness=2,
                groundtruth_box_visualization_color='black',
            )
            # print(person.shape[0])
            # print(person)

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            cv2.putText(image_np, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

            cv2.imshow("frame", image_np)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            # time.sleep(2)
            # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()



