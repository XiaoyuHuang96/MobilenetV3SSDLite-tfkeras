import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import pathlib

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './trained_models/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('./imgs/test')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
# TEST_IMAGE_PATHS


weights_path = './pretrained_weights/ssd_mobilenet_v3_small_coco_2020_01_14/covert_from_tf_small0503/frozen_inference_graph.pb'
# model = tf.saved_model.load(model_name)
# model = model.signatures['serving_default']
# model.load_weights(model_name)
# ckpt = tf.train.Checkpoint(model = model)
# ckpt.restore(tf.train.latest_checkpoint(weights_path))
# model.training = False

# def load_model(model_name):
#   base_url = 'http://download.tensorflow.org/models/object_detection/'
#   model_file = model_name + '.tar.gz'
#   model_dir = tf.keras.utils.get_file(
#     fname=model_name, 
#     origin=base_url + model_file,
#     untar=True)

#   model_dir = pathlib.Path(model_dir)/"saved_model"

#   model = tf.saved_model.load_v2(str(model_dir))
#   model = model.signatures['serving_default']

#   return model


# with tf.Session() as sess:
#     new_saver = tf.train.import_meta_graph('./pretrained_weights/ssd_mobilenet_v3_small_coco_2020_01_14/model.ckpt.meta')
#     new_saver.restore(sess, tf.train.latest_checkpoint('./pretrained_weights/ssd_mobilenet_v3_small_coco_2020_01_14'))
#     model=sess
# model = load_model('ssd_mobilenet_v3_small_coco_2020_01_14')

# model = tf.saved_model.load_v2(model_name)
# model = model.signatures['serving_default']

# print(model.inputs)

# Size, in inches, of the output images.
IMAGE_SIZE = (24, 24)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      print("image",image)

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, axis=0)})

      before_postprocess = tf.get_default_graph().get_tensor_by_name('Concatenate/concat:0')
      print("anchors",before_postprocess,before_postprocess.eval())
      

      before_postprocess = tf.get_default_graph().get_tensor_by_name('concat:0')
      print("concat",before_postprocess)
      value = sess.run(before_postprocess, feed_dict={image_tensor: np.expand_dims(image, axis=0)})
      print("concat",value,value.shape)

      # before_input = tf.get_default_graph().get_tensor_by_name('FeatureExtractor/MobilenetV3/MobilenetV3/input:0')
      # print("inputimg",before_input)
      # value = sess.run(before_input, feed_dict={image_tensor: np.expand_dims(image, axis=0)})
      # print("inputimg",value,value.shape)

      # after_expconv = tf.get_default_graph().get_tensor_by_name('FeatureExtractor/MobilenetV3/expanded_conv/squeeze_excite/mul:0')
      # print("after_expconv_se",after_expconv)
      # value = sess.run(after_expconv, feed_dict={image_tensor: np.expand_dims(image, axis=0)})
      # print("after_expconv_se",value,value.shape)
      

      # afterbackbone = tf.get_default_graph().get_tensor_by_name('FeatureExtractor/MobilenetV3/expanded_conv_8/expansion_output:0')
      # print("afterbackbone_0",afterbackbone)
      # value = sess.run(afterbackbone, feed_dict={image_tensor: np.expand_dims(image, axis=0)})
      # print("afterbackbone_0",value,value.shape)

      # afterbackbone = tf.get_default_graph().get_tensor_by_name('FeatureExtractor/MobilenetV3/Conv_1/hard_swish/mul_1:0')
      # print("afterbackbone_1",afterbackbone)
      # value = sess.run(afterbackbone, feed_dict={image_tensor: np.expand_dims(image, axis=0)})
      # print("afterbackbone_1",value,value.shape)

      

      # before_postprocess = tf.get_default_graph().get_tensor_by_name('BoxPredictor_0/Reshape:0')
      # print("BoxPredictor_0/Reshape/(Reshape)",before_postprocess)
      # value = sess.run(before_postprocess, feed_dict={image_tensor: np.expand_dims(image, axis=0)})
      # print("BoxPredictor_0/Reshape/(Reshape)",value,value.shape)

      
      # before_postprocess = tf.get_default_graph().get_tensor_by_name('concat:0')
      # print("concat",before_postprocess,before_postprocess.eval())

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


def run_detection(path_to_frozen_graph, title):
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  plt.figure(figsize=IMAGE_SIZE)
  
  for i, image_path in enumerate(TEST_IMAGE_PATHS):
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image = image.resize((320,320),Image.ANTIALIAS)
    image_np = load_image_into_numpy_array(image)

    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    print("output",output_dict)

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)

    plt.subplot(1, 2, i+1)
    plt.imshow(image_np)
    plt.title(title)
  
  plt.savefig('./imgs/detections_test/tf_result.jpg')


run_detection(weights_path, "ssd_mobilenet_v3_small")