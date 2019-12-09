import tensorflow as tf
import cv2
import json
import os
import numpy as np
from keras.backend import set_session
from keras.models import load_model
from ai_tree.yolov3_expe.utils.utils import get_yolo_boxes, makedirs
from ai_tree.yolov3_expe.utils.bbox import draw_boxes

def preload_model(mod_paths):
    infer_models = {}
    for t, p in mod_paths.items():
        if t == 'yolov3_leaf':
            infer_models['yolov3_leaf'] = load_mod(p)
        elif t == 'yolov3_tree':
            infer_models['yolov3_tree'] = load_mod(p)
        elif t == 'clr_leaf':
            infer_models['clr_leaf'] = load_mod(p)
        elif t == 'clr_tree':
            infer_models['clr_tree'] = load_mod(p)

    return infer_models


def load_mod(path):
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with graph.as_default():
        set_session(sess)
        mod = load_model(path)
    return [graph, sess, mod]


def yolo_predict(config_path, input_path, output_path, infer_model):
    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Set some parameter
    ###############################
    net_h, net_w = 416, 416  # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Predict bounding boxes
    ###############################

    # do detection on an image or a set of images
    image_paths = []

    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

    # the main loop
    for image_path in image_paths:
        image = cv2.imread(image_path)
        print(image_path)
        graph = infer_model[0]
        sess = infer_model[1]
        model = infer_model[2]
        with graph.as_default():
            set_session(sess)
            # predict the bounding boxes
            boxes = get_yolo_boxes(model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]
            print("model ready and run")

        # draw bounding boxes on the image using labels
        draw_boxes(image, boxes, config['model']['labels'], obj_thresh)

        # write the image with bounding boxes to file
        cv2.imwrite(output_path + image_path.split('/')[-1], np.uint8(image))

def clr_pred(input_path, conf_path, model):
    graph = model[0]
    sess = model[1]
    mod = model[2]

    with open(conf_path) as config_buffer:
        config = json.load(config_buffer)
    net_h, net_w = config['model']['input_size'], config['model']['input_size']
    pred_class = config['model']['labels']

    img = cv2.imread(input_path)
    imgs = np.zeros((1, net_h, net_w, 3))
    img = np.resize(img, (net_h, net_w, 3))
    imgs[0] = img
    with graph.as_default():
        set_session(sess)
        y_pred = mod.predict(imgs)

    return pred_class[np.argmax(y_pred)]










