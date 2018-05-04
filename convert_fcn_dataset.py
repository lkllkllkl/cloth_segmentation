#!/usr/bin/env python3
import logging
import os

import cv2
import numpy as np
import tensorflow as tf
import six
import collections
from vgg import vgg_16


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')

FLAGS = flags.FLAGS

classes = ['background', 'accessories', 'bag', 'belt', 'blazer', 
            'blouse', 'bodysuit', 'boots', 'bra', 'bracelet',
            'cape', 'cardigan', 'clogs', 'coat', 'dress', 
            'earrings', 'flats', 'glasses', 'gloves', 'hair', 
            'hat', 'heels', 'hoodie', 'intimate', 'jacket', 
            'jeans', 'jumper', 'leggings', 'loafers', 'necklace', 
            'panties', 'pants', 'pumps', 'purse', 'ring', 
            'romper', 'sandals', 'scarf', 'shirt', 'shoes', 
            'shorts', 'skin', 'skirt', 'sneakers','socks',
            'stockings', 'suit', 'sunglasses', 'sweater','sweatshirt',
            'swimwear', 't-shirt', 'tie', 'tights', 'top', 
            'vest', 'wallet', 'watch', 'wedges']


colormap = [[0, 0, 0], [165, 60, 45], [77, 164, 38], [202, 109, 103], [24, 18, 141],
            [37, 49, 58], [48, 36, 27], [187, 34, 238], [29, 128, 11], [109, 103, 226], 
            [19, 163, 16], [44, 203, 131], [222, 39, 93], [214, 118, 46], [35, 5, 110],
            [123, 156, 117], [46, 247, 79], [217, 184, 223], [30, 150, 240], [63, 111, 147], 
            [114, 213, 96], [31, 172, 214], [203, 131, 77], [25, 40, 115], [113, 191, 122], 
            [23, 251, 167], [68, 221, 17], [148, 196, 232], [73, 76, 142], [60, 45, 225],
            [157, 139, 253], [92, 239, 158], [52, 124, 178], [96, 72, 54], [190, 100, 160],
            [49, 58, 1], [32, 194, 188], [105, 15, 75], [254, 233, 26], [218, 206, 197], 
            [160, 205, 175], [238, 136, 187], [232, 4, 88], [185, 245, 35], [153, 51, 102], 
            [127, 244, 13], [124, 178, 91], [41, 137, 209], [253, 211, 52], [175, 25, 40],
            [229, 193, 166], [147, 174, 3], [84, 63, 111], [250, 145, 130], [215, 140, 20], 
            [20, 185, 245], [39, 93, 6], [174, 3, 66], [179, 113, 191]]


cm2lbl = np.zeros(256**3)
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

_IMAGE_FORMAT_MAP = {
    'jpg': 'jpeg',
    'jpeg': 'jpeg',
    'png': 'png',
}

def image2label(im):
    data = im.astype('int32')
    # cv2.imread. default channel layout is BGR
    idx = (data[:, :, 2] * 256 + data[:, :, 1]) * 256 + data[:, :, 0]
    return np.array(cm2lbl[idx])


def dict_to_tf_example(data, label):
    with open(data, 'rb') as inf:
        encoded_data = inf.read()
    img_label = cv2.imread(label)
    img_mask = image2label(img_label)
    encoded_label = img_mask.astype(np.uint8).tobytes()

    height, width = img_label.shape[0], img_label.shape[1]
    if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
        # 保证最后随机裁剪的尺寸
        return None

    # Your code here, fill the dict
    feature_dict = {
        'image/height': _int64_list_feature(height),
        'image/width': _int64_list_feature(width),
        'image/filename': _bytes_list_feature(os.path.basename(data)[:-4]),
        'image/encoded': _bytes_list_feature(encoded_data),
        'image/label': _bytes_list_feature(encoded_label),
        'image/format': _bytes_list_feature(_IMAGE_FORMAT_MAP[os.path.basename(data)[-3:]]),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example



def _int64_list_feature(values):
  """Returns a TF-Feature of int64_list.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, collections.Iterable):
    values = [values]

  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  def norm2bytes(value):
      return value.encode() if isinstance(value, str) and six.PY3 else value

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


def create_tf_record(output_filename, file_pars):
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for (data, label) in file_pars:
            print(data)
            example = dict_to_tf_example(data, label)
            if not example is None:
                tfrecord_writer.write(example.SerializeToString())
        
    

def read_images_names(root, train=True):
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/', 'train.txt' if train else 'val.txt')

    with open(txt_fname, 'r') as f:
        images = f.read().split()

    data = []
    label = []
    for fname in images:
        data.append('%s/JPEGImages/%s.jpg' % (root, fname))
        label.append('%s/SegmentationClass/%s.png' % (root, fname))
    return zip(data, label)


def main(_):
    logging.info('Prepare dataset file names')

    train_output_path = os.path.join(FLAGS.output_dir, 'fcn_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'fcn_val.record')

    train_files = read_images_names(FLAGS.data_dir, True)
    val_files = read_images_names(FLAGS.data_dir, False)
    create_tf_record(train_output_path, train_files)
    create_tf_record(val_output_path, val_files)


if __name__ == '__main__':
    tf.app.run()
