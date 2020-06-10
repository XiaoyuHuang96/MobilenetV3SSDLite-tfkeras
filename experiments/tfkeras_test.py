from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
from losses.keras_ssd_loss import SSDLoss
# from models.keras_mobilenet_v2_ssdlite import mobilenet_v2_ssd
from models.mbv3ssdlite_model import MobileNetV3SSDLite
import os
import tensorflow as tf
from ckpt2h5 import load_weights_from_tf_checkpoint,MyCallbacks
from tensorflow.keras.utils import plot_model
import h5py
import tensorflow.keras.backend as K
import matplotlib.image as mpimg
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from tqdm import trange,tqdm
from layers.DecodeDetectionsLayer import DecodeDetections

# model config
batch_size = 16
image_size = (320, 320, 3)
n_classes = 90
mode = 'inference'
l2_regularization = 0.0005
min_scale = 0.1 #None
max_scale = 0.95 #None
scales = [0.1, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0] 
aspect_ratios_global = None #[1.0, 2.0, 0.5, 3.0, 1.0 / 3.0] # None
aspect_ratios_per_layer = [[1.0, 2.0, 0.5], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]]
                           # None 
                           #[[1.0, 2.0, 0.5], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           #[1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5], [1.0, 2.0, 0.5]]
two_boxes_for_ar1 = True
steps = None #[8, 16, 32, 64, 100, 300]
offsets = None #[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
clip_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
coords = 'centroids'
normalize_coords = True
subtract_mean = [123, 117, 104]
divide_by_stddev = 128
swap_channels = None
confidence_thresh = 0.2 #0.65
iou_threshold = 0.45
top_k = 200
nms_max_output_size = 400
return_predictor_sizes = False
model_type="small_extractor"
width_multiplier=1.0
model_name="FeatureExtractor"#"Mobilenetv3_Large"
divisible_by=8

K.set_learning_phase(0)

model = MobileNetV3SSDLite(image_size, n_classes, mode, l2_regularization, min_scale, max_scale, scales,
                         aspect_ratios_global, aspect_ratios_per_layer, two_boxes_for_ar1, steps,
                         offsets, clip_boxes, variances, coords, normalize_coords, subtract_mean,
                         divide_by_stddev, swap_channels, confidence_thresh, iou_threshold, top_k,
                         nms_max_output_size, return_predictor_sizes,model_type,width_multiplier,model_name,divisible_by)

input_tensor = tf.keras.layers.Input(shape=image_size)
output_tensor = model(input_tensor)

model = tf.keras.Model(inputs=[model.input],outputs=[model.output])

print(model.summary())
# FeatureExtractor
print(model.layers[1].summary())
print(model.layers[1].layers[3].summary())
# MobileNetv3
print(model.layers[1].layers[3].layers[0].summary())

# 2: Load the trained weights into the model.
weights_path = './pretrained_weights/tf_convert_mbv3ssdlite_small_0503.h5'

initial_weights = [layer.get_weights() for layer in model.layers]

model.load_weights("./logs/0506_1/mbv3_ssdlite_coco_06_loss-8.6478_val_loss-10.4478.h5", by_name=True)

for layer, initial in zip(model.layers, initial_weights):
    weights = layer.get_weights()
    
    print(layer.name)
    print("len(weights)",len(weights))

    # print(initial)
    if weights and all(tf.nest.map_structure(np.array_equal, weights, initial)):
        print(f'first loaded contained no weights for layer {layer.name}!')

# h5 weights loaded successfully


# save weights
# model.save_weights("./ourmodels/0505.h5")

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './testfile/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# print("category_index",category_index)

for layer, initial in zip(model.layers, loaded_weights):
    weights = layer.get_weights()
    print(layer.name)
    # print(initial)
    if weights and all(tf.nest.map_structure(np.array_equal, weights, initial)):
        print(f'second loaded contained no weights for layer {layer.name}!')


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  imgdata = np.array(image.getdata())
#   print("imgdata",imgdata.shape)
  if(len(imgdata.shape)==1):
      return imgdata.reshape((im_height, im_width, 1)).astype(np.uint8).repeat(3,axis=2)
  return imgdata.reshape(
      (im_height, im_width, 3)).astype(np.uint8)


coco_root = '/home/huangxiaoyu/datasets/coco/'

gt_json_path = os.path.join(coco_root,"annotations","instances_val2014.json")
coco_gt = COCO(gt_json_path)
# image_ids = coco_gt.getImgIds()

fn = open('./testfile/mscoco_val_ids.txt','r')
image_ids = []
for line in fn:
    line = line.split()[0]
    image_ids.append(int(line))

# We'll only load one image in this example.
img_dir = os.path.join(coco_root,"images","val2014")
#img_dir = '/home/huangxiaoyu/datasets/coco/images/val2017'
#orig_images = []
#input_images = []

results = []
processed_image_ids = []

# for imgfn in os.listdir(img_dir):
#     img_path = os.path.join(img_dir, imgfn)
for image_id in tqdm(image_ids):
    image_info = coco_gt.loadImgs(image_id)[0]

    processed_image_ids.append(image_id)

    img_path = os.path.join(img_dir, image_info['file_name'])
    # print(img_path)
    orig_images = [] # Store the images here.
    input_images = [] # Store resized versions of the images here.

    orig_images.append(imread(img_path))
    
    
    # keras preprocess
    img = image.load_img(img_path, target_size=(image_size[0], image_size[1]))
    img = image.img_to_array(img)
    input_images.append(img)
    
    # tf preprocess
    # image = Image.open(img_path)
    # image = image.resize((image_size[0],image_size[1]),Image.ANTIALIAS)
    # image_np = load_image_into_numpy_array(image)
    # input_images.append(image_np)


    input_images = np.array(input_images,dtype=np.float32)
    y_pred = model.predict(input_images)

    print("predict end.")

    confidence_threshold = confidence_thresh

    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print("y_pred_thresh[0]",y_pred_thresh[0])

    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2] * orig_images[0].shape[1] / image_size[0]
        ymin = box[3] * orig_images[0].shape[0] / image_size[1]
        xmax = box[4] * orig_images[0].shape[1] / image_size[1]
        ymax = box[5] * orig_images[0].shape[0] / image_size[0]
        
       
        if(int(box[0]) not in category_index.keys()):
            continue
        clsname = category_index[int(box[0])]['name']

        image_result = {
            'image_id': image_id,
            'category_id': int(box[0]),
            'score': float(box[1]),
            'bbox': [xmin, ymin, xmax-xmin, ymax-ymin],
        }

        results.append(image_result)
        
if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

# write output
json.dump(results, open('tfkeras_{}model_preds.json'.format(model_type[:model_type.find('_')]), 'w'), indent=4)

def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


_eval(coco_gt, image_ids, 'tfkeras_{}model_preds.json'.format(model_type[:model_type.find('_')]))