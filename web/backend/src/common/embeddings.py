"""
Exports the embeddings and labels of a directory of images as numpy arrays.
Typicall usage expect the image directory to be of the openface/facenet form and
the images to be aligned. Simply point to your model and your image directory:
    python facenet/contributed/export_embeddings.py ~/models/facenet/20170216-091149/ ~/datasets/lfw/mylfw
Output:
embeddings.npy -- Embeddings as np array, Use --embeddings_name to change name
labels.npy -- Integer labels as np array, Use --labels_name to change name
label_strings.npy -- Strings from folders names, --labels_strings_name to change name
Use --image_batch to dictacte how many images to load in memory at a time.
If your images aren't already pre-aligned, use --is_aligned False
I started with compare.py from David Sandberg, and modified it to export
the embeddings. The image loading is done use the facenet library if the image
is pre-aligned. If the image isn't pre-aligned, I use the compare.py function.
I've found working with the embeddings useful for classifications models.
Charles Jekel 2017
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import __main__

import time
#from scipy import misc
#import tensorflow as tf
import numpy as np
import sys
import os
import argparse
#import facenet
#import facenet
#import align.detect_face
import glob

import tensorflow as tf

try:
    import tensorflow.contrib.tensorrt as trt # must import this although we will not use it explicitly
except:
    print("Tensor RT not loaded")

from tensorflow.python.platform import gfile
#from PIL import Image



from six.moves import xrange

class FN_Embeddings(object):

    # TODO make this dynamic
    TENSORRT_MODEL_PATH = '/home/karlo/web/backend/20180402-114759/TensorRT_model.pb'

    sess = None

    images_placeholder = None
    embeddings = None
    phase_train_placeholder = None

    def read_pb_graph(self, model):
        with gfile.FastGFile(model,'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def

    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y  

    def return_emb(self, image):
        image = self.prewhiten(image)

        image_list = [image]
        feed_dict = {self.images_placeholder: [image_list], self.phase_train_placeholder: False}

        start_time = time.time()

        out_pred = self.sess.run(self.embeddings, feed_dict=feed_dict)

        print("predicted in: " + str(time.time() - start_time))
        
        print(out_pred[0])

        return out_pred[0] # First element since only one image is being passes at a time

    def face_distance(self, embedding_reference, embedding_to_compare):
        """
        Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
        for each comparison face. The distance tells you how similar the faces are.
        :param faces: List of face encodings to compare
        :param face_to_compare: A face encoding to compare against
        :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
        """

        dist = np.sqrt(np.sum(np.square(np.subtract(embedding_reference, embedding_to_compare))))
        return dist

        #import numpy as np
        #if len(face_encodings) == 0:
        #    return np.empty((0))

        #return 1/np.linalg.norm(face_encodings - face_to_compare, axis=1)
        #return np.sum(face_encodings*face_to_compare,axis=1)


    def __init__(self):
        if os.name != 'nt':
            self.sess =  tf.Session(config=tf.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3), log_device_placement=True))
            trt_graph = self.read_pb_graph(self.TENSORRT_MODEL_PATH)
            
            tf.import_graph_def(trt_graph, input_map=None, name='')
            
            self.images_placeholder = self.sess.graph.get_tensor_by_name("batch_join:0") #jjia changed 2018/01/21        
            self.embeddings = self.sess.graph.get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = self.sess.graph.get_tensor_by_name("phase_train:0")
        else:
            print("Running on windows") # TODO globaly disable any tensorflow process

        return
              



