from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
# from imageio import imread
import numpy as np
# from matplotlib import pyplot as plt
from losses.keras_ssd_loss import SSDLoss
# from models.keras_mobilenet_v2_ssdlite import mobilenet_v2_ssd
from models.mbv3ssdlite_model import MobileNetV3SSDLite
import os
import tensorflow as tf
from ckpt2h5 import load_weights_from_tf_checkpoint,MyCallbacks
from tensorflow.keras.utils import plot_model
import h5py
import tensorflow.keras.backend as K
# import matplotlib.image as mpimg
# from PIL import Image
# from object_detection.utils import ops as utils_ops
# from object_detection.utils import label_map_util
# from layers.DecodeDetectionsLayer import DecodeDetections
# from utils.ssd_output_decoder import decode_detections,decode_detections_fast

# model config
batch_size = 32
image_size = (320, 320, 3)
n_classes = 90
mode = 'inference_fast'
l2_regularization = 0.0004
min_scale = 0.2 #None
max_scale = 0.95 #None
scales =  [0.1, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0]
aspect_ratios_global = None #[1.0, 2.0, 0.5, 3.0, 1.0 / 3.0] # None
aspect_ratios_per_layer = [[1.01, 2.0, 0.5], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]]
                           # None 
                           #[[1.0, 2.0, 0.5], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           #[1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5], [1.0, 2.0, 0.5]]
two_boxes_for_ar1 = True
steps = None #[16, 32, 64, 107, 160, 320]
offsets = None #[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
clip_boxes = False #True
variances = [1.0,1.0,1.0,1.0]#[0.1, 0.1, 0.1, 0.1]#[0.1, 0.1, 0.2, 0.2]
coords = 'centroids'
normalize_coords = True
subtract_mean =  [0,0,0] #[123, 117, 104]
divide_by_stddev =  1 #28
swap_channels =  None
confidence_thresh = 0.1 # 0.65
iou_threshold = 0.6
top_k = 100 #200
nms_max_output_size = 100 #400
return_predictor_sizes = False
model_type="large_extractor"
width_multiplier=1.0
model_name="FeatureExtractor"#"Mobilenetv3_Large"
divisible_by=8

K.clear_session()
K.set_learning_phase(0)

model = MobileNetV3SSDLite(image_size, n_classes, mode, l2_regularization, min_scale, max_scale, scales,
                         aspect_ratios_global, aspect_ratios_per_layer, two_boxes_for_ar1, steps,
                         offsets, clip_boxes, variances, coords, normalize_coords, subtract_mean,
                         divide_by_stddev, swap_channels, confidence_thresh, iou_threshold, top_k,
                         nms_max_output_size, return_predictor_sizes,model_type,width_multiplier,model_name,divisible_by)
input_tensor = tf.keras.layers.Input(shape=image_size)
output_tensor = model(input_tensor)
# ,input_shape=image_size

model = tf.keras.Model(inputs=[model.input],outputs=[model.output],trainable=False)

weights_path = './ourmodels/loaded_large_0528_weights.h5'

model.load_weights(weights_path,by_name=True)

def load_pb(pb_file_path):
    sess = tf.Session()
    with tf.gfile.FastGFile(pb_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        print(graph_def.node[0])
        print(graph_def.node[-1])
    return sess

def convertsess2tflite(pb_path = './ourmodels/convert_large_0609.pb'):
  # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  out_prefix = 'output_'
  out_nodes = []
  for i in range(len(model.outputs)):
      print("model.output",i,model.output[i].name,out_prefix + str(i + 1))
      out_nodes.append(out_prefix + str(i + 1))
      tf.identity(model.output[i],out_prefix + str(i + 1))
  from tensorflow.python.keras.backend import set_session
  
  K.set_learning_phase(0)
  sess = K.get_session()
  set_session(sess)

  # sess = load_pb(pb_path)

  tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
  indx = 0
  for tensor_name in tensor_name_list:
    if 'SSDLite' in tensor_name:
      print(indx,tensor_name,'\n')

    indx += 1 

  # sess.run(tf.global_variables_initializer())
  input_tensors = sess.graph.get_tensor_by_name('input_1:0')
  output_tensors = sess.graph.get_tensor_by_name('output_1:0')
  # output_tensors = sess.graph.get_tensor_by_name('SSDLite/ssd_decoded_predictions/strided_slice:0')
  # SSDLite/ssd_decoded_predictions/loop_over_batch/TensorArrayStack/TensorArrayGatherV3:0

  # input_shape = {"input_1" : [1, 320, 320, 3]}
  converter = tf.lite.TFLiteConverter.from_session(sess, [input_tensors], [output_tensors])
  tflite_model = converter.convert()
  open("./ourmodels/convert_large_0609.tflite", "wb").write(tflite_model)

def convertkeras2pb(output_dir = './ourmodels/',output_graph_name = 'convert_large_0609.pb'):
  from keras2pb import h5_to_pb
  h5_to_pb(model,output_dir = output_dir,model_name = output_graph_name)
  print('model saved')


def convertpb2tflite(graph_def_file = './ourmodels/convert_large.pb'):
  # Converting a GraphDef from file.
  input_arrays = ["input_1"]
  output_arrays = ["output_1"]
  input_shape = {"input_1" : [1, 320, 320, 3]}
  converter = tf.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file, input_arrays, output_arrays,input_shape)
  converter.experimental_new_converter = True
  tflite_model = converter.convert()
  open("./ourmodels/convert_large_inferfast/convert_large_inferfast_0609.tflite", "wb").write(tflite_model)


def convertckpt2tflite(ckpt_path):
  sess = K.get_session()
  from tensorflow.python.framework import graph_util,graph_io
  init_graph = sess.graph.as_graph_def()
  main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)

def tf2pb2tflite(graph_def_file = './ourmodels/convert_large.pb'):
  # Converting a GraphDef from file.
  input_arrays = ["input_1"]
  output_arrays = ["output_1"]
  input_shape = {"input_1" : [1, 320, 320, 3]}
  converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file, input_arrays, output_arrays,input_shape)
  tflite_model = converter.convert()
  open("./ourmodels/convert_large.tflite", "wb").write(tflite_model)



def tf2keras2tflite():
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  open("./ourmodels/convert_large.tflite", "wb").write(tflite_model)


if __name__ == "__main__":
  # convertkeras2pb(output_dir='./ourmodels/convert_large_inferfast',output_graph_name = 'convert_large_inferfast_0609.pb')
  convertpb2tflite(graph_def_file='./ourmodels/convert_large_inferfast/convert_large_inferfast_0609.pb')
  # convertsess2tflite()
  # tf2keras2tflite()
  # tf2pb2tflite()
