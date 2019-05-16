#!/usr/bin/env python
# coding: utf-8

# # The Nature Conservancy Fisheries Monitoring

# https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring

import os
import json
from glob import glob
import sys
import cv2
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
import keras.backend as K

# # Загружаем разметку

#CUDA_VISIBLE_DEVICES=''
# TODO: скачайте данные и сохраните в директорию:
TRAIN_PREFIX = './data/fish/train'
VALIDATION_PREFIX = './data/fish/test_stg1'
ORIGINAL_IMG_HEIGHT = 750
ORIGINAL_IMG_WIDTH = 1200
IMG_HEIGHT = 468#int(750/1.6)
IMG_WIDTH = 752#int(1200/1.6)

ANCHOR_WIDTH = 100#150.
ANCHOR_HEIGHT = 100#150. 

label_encoder = dict()
str_labels = []

FEATURE_SHAPE = (14,23)

GRID_STEP_H = IMG_HEIGHT / FEATURE_SHAPE[0]
GRID_STEP_W = IMG_WIDTH / FEATURE_SHAPE[1]

ANCHOR_CENTERS = np.mgrid[GRID_STEP_H/2:IMG_HEIGHT:GRID_STEP_H,
                          GRID_STEP_W/2:IMG_WIDTH:GRID_STEP_W]

def load_boxes(path_mask = './data/fish/boxes/*.json', prefix=TRAIN_PREFIX):
    boxes = dict()
    for path in glob(path_mask):
        label = os.path.basename(path).split('_', 1)[0]
        with open(path) as src:
            boxes[label] = json.load(src)
            for annotation in boxes[label]:
                basename = os.path.basename(annotation['filename'])
                annotation['filename'] = os.path.join(TRAIN_PREFIX, label.upper(), basename)
            for annotation in boxes[label]:
                for rect in annotation['annotations']:
                    rect['x'] += rect['width'] / 2
                    rect['y'] += rect['height'] / 2
    return boxes

def load_valid_boxes():
    return load_boxes(path_mask = './data/fish/validation_boxes/*.json',
		      prefix = VALIDATION_PREFIX)

def load_NoFiles():
    files = list()
    files = [file for file in glob('./data/fish/train/NoF/*.jpg')]
    return files

def draw_boxes(annotation, rectangles=None, image_size=None):
    
    def _draw(img, rectangles, scale_x, scale_y, color=(0, 255, 0)):
        for rect in rectangles:
            pt1 = (int((rect['x'] - rect['width'] / 2) * scale_x),
                   int((rect['y'] - rect['height'] / 2) * scale_y))
            pt2 = (int((rect['x'] + rect['width'] / 2) * scale_x),
                   int((rect['y'] + rect['height'] / 2) * scale_y))
            img = cv2.rectangle(img.copy(), pt1, pt2, 
                                color=color, thickness=4)
        return img
    
    scale_x, scale_y = 1., 1. 
    
    img = cv2.imread(annotation['filename'], cv2.IMREAD_COLOR)[...,::-1]
    if image_size is not None:
        scale_x = 1. * image_size[0] / img.shape[1]
        scale_y = 1. * image_size[1] / img.shape[0]
        img = cv2.resize(img, image_size)
        
    img = _draw(img, annotation['annotations'], scale_x, scale_y)
    
    if rectangles is not None:
        img = _draw(img, rectangles, 1., 1., (255, 0, 0))

    return img

def make_labels(aux_lb = []):
	labels = []
	for path in glob('./data/fish/boxes/*.json'):
		labels.append(os.path.basename(path).split('_', 1)[0].upper())
	for str1 in aux_lb:
		labels.append(str1)
	labels_cat = pd.get_dummies(labels)
	labels_cat = labels_cat.sort_values(by=labels[0], ascending=False).values.tolist()
	label_dict = {dkey: dval for dkey, dval in zip(labels, labels_cat)}
	global label_encoder;
	label_encoder = label_dict
	global str_labels
	str_labels = labels
	return label_dict, labels

def get_feature_tensor():
	features = keras.applications.vgg16.VGG16(include_top=False,
					          weights='imagenet',
					          input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
	output = features.layers[-1].output
	global FEATURE_SHAPE
	FEATURE_SHAPE = (output.shape[1].value,
                 output.shape[2].value)
	global GRID_STEP_H
	GRID_STEP_H = IMG_HEIGHT / FEATURE_SHAPE[0]
	global GRID_STEP_W
	GRID_STEP_W = IMG_WIDTH / FEATURE_SHAPE[1]
	global ANCHOR_CENTERS
	ANCHOR_CENTERS = np.mgrid[GRID_STEP_H/2:IMG_HEIGHT:GRID_STEP_H,
		                  GRID_STEP_W/2:IMG_WIDTH:GRID_STEP_W]
	return features

def set_tensor_shape():
	global FEATURE_SHAPE
	global GRID_STEP_H
	GRID_STEP_H = IMG_HEIGHT / FEATURE_SHAPE[0]
	global GRID_STEP_W
	GRID_STEP_W = IMG_WIDTH / FEATURE_SHAPE[1]
	global ANCHOR_CENTERS
	ANCHOR_CENTERS = np.mgrid[GRID_STEP_H/2:IMG_HEIGHT:GRID_STEP_H,
		                  GRID_STEP_W/2:IMG_WIDTH:GRID_STEP_W]
	return 

def iou(rect, x_scale, y_scale, anchor_x, anchor_y,
        anchor_w=ANCHOR_WIDTH, anchor_h=ANCHOR_HEIGHT):
    
    rect_x1 = (rect['x'] - rect['width'] / 2) * x_scale
    rect_x2 = (rect['x'] + rect['width'] / 2) * x_scale
    
    rect_y1 = (rect['y'] - rect['height'] / 2) * y_scale
    rect_y2 = (rect['y'] + rect['height'] / 2) * y_scale
    
    anch_x1, anch_x2 = anchor_x - anchor_w / 2, anchor_x + anchor_w / 2
    anch_y1, anch_y2 = anchor_y - anchor_h / 2, anchor_y + anchor_h / 2
    
    dx = (min(rect_x2, anch_x2) - max(rect_x1, anch_x1))
    dy = (min(rect_y2, anch_y2) - max(rect_y1, anch_y1))
    
    intersection = dx * dy if (dx > 0 and dy > 0) else 0.
    
    anch_square = (anch_x2 - anch_x1) * (anch_y2 - anch_y1)
    rect_square = (rect_x2 - rect_x1) * (rect_y2 - rect_y1)
    union = anch_square + rect_square - intersection
    
    return intersection / union

def encode_anchors(annotation, img_shape, iou_thr=0.):
    encoded = np.zeros(shape=(FEATURE_SHAPE[0],
                              FEATURE_SHAPE[1], 5 + len(str_labels)), 
                              dtype=np.float32)
    x_scale = 1. * img_shape[1]/ORIGINAL_IMG_WIDTH
    y_scale = 1. * img_shape[0]/ORIGINAL_IMG_HEIGHT
    label = annotation['filename'].split("train/", 1)[1]
    label = label.split("img", 1)[0]
    label = label.split("/", 1)[0]
    for rect in annotation['annotations']:
        scores = []
        for row in range(FEATURE_SHAPE[0]):
            for col in range(FEATURE_SHAPE[1]):
                anchor_x = ANCHOR_CENTERS[1, row, col]
                anchor_y = ANCHOR_CENTERS[0, row, col]
                score = iou(rect, x_scale, y_scale, anchor_x, anchor_y)
                scores.append((score, anchor_x, anchor_y, row, col))
        
        scores = sorted(scores, reverse=True)
        if scores[0][0] < iou_thr:
            scores = [scores[0]]  # default anchor
        else:
            scores = [e for e in scores if e[0] > iou_thr]

        for score, anchor_x, anchor_y, row, col in scores:
            dx = (anchor_x - rect['x'] * x_scale) / ANCHOR_WIDTH
            dy = (anchor_y - rect['y'] * y_scale) / ANCHOR_HEIGHT
            dw = (ANCHOR_WIDTH - rect['width'] * x_scale) / ANCHOR_WIDTH
            dh = (ANCHOR_HEIGHT - rect['height'] * y_scale) / ANCHOR_HEIGHT
            encoded[row, col] = [1., dx, dy, dw, dh] + label_encoder[label]
            #encoded[row, col] = [1., dx, dy, dw, dh] + label_encoder[label]
    return encoded

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def decode_prediction(prediction, conf_thr=0.1):
    rectangles = []
    conf = 0
    maxcol = 0
    maxrow = 0
    for row in range(FEATURE_SHAPE[0]):
        for col in range(FEATURE_SHAPE[1]):
            class_probabilities = list(range(len(str_labels)))
            logit =  prediction[row, col][0]
            new_conf = _sigmoid(logit)
            if (new_conf > conf):
                conf = new_conf
                maxcol = col
                maxrow = row
    logit, dx, dy, dw, dh =  prediction[maxrow, maxcol][0:5]
    conf = _sigmoid(logit)
    class_logits =  prediction[maxrow, maxcol][5:]
    class_probabilities = _sigmoid(class_logits)
    class_probab_NoF = (1 - conf)
    class_probab_Other = int(conf > 0.5)*(1 - max(class_probabilities))
    class_norma = sum(class_probabilities) + class_probab_NoF + class_probab_Other
    class_probabilities = class_probabilities/class_norma
    class_probab_NoF = class_probab_NoF/class_norma
    class_probab_Other = class_probab_Other/class_norma
    if ((class_probab_Other <= 0.5) & (class_probab_NoF <= 0.5)):
        class_label = [int(e > 0.5) for e in class_probabilities]
        class_name = [key for key in label_encoder.keys() if (label_encoder[key] == class_label)]
        class_label.append(0)
        class_label.append(0)
    elif (class_probab_Other > 0.5):
        class_label = [0, 0, 0, 0, 0, 0, 0, 1]
        class_name = "OTHER"
    elif (class_probab_NoF > 0.5):
        class_label = [0, 0, 0, 0, 0, 0, 1, 0]
        class_name = "NoF"
    class_probabilities = np.append(class_probabilities, class_probab_NoF)
    class_probabilities = np.append(class_probabilities, class_probab_Other)
    class_probabilities = np.append(class_probabilities[:4] , [class_probabilities[6:] , class_probabilities[4:6]])
    anchor_x = ANCHOR_CENTERS[1, maxrow, maxcol]
    anchor_y = ANCHOR_CENTERS[0, maxrow, maxcol]
    rectangles.append({'x': anchor_x - dx * ANCHOR_WIDTH,
	           'y': anchor_y - dy * ANCHOR_HEIGHT,
	           'width': ANCHOR_WIDTH - dw * ANCHOR_WIDTH,
	           'height': ANCHOR_HEIGHT - dh * ANCHOR_HEIGHT,
	           'conf': conf,
	           'class_probab': class_probabilities,
	           'class_name': class_name})
    return rectangles


def load_img(path, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    img = cv2.imread(path, cv2.IMREAD_COLOR)[...,::-1]
    img_shape = img.shape
    img_resized = cv2.resize(img, target_size)
    return img_shape, keras.applications.vgg16.preprocess_input(img_resized.astype(np.float32))

