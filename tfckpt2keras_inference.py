from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
from losses.keras_ssd_loss import SSDLoss
from models.mbv3ssdlite_model import MobileNetV3SSDLite
import os
import tensorflow as tf
import h5py
import tensorflow.keras.backend as K
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



# model config
batch_size = 32
image_size = (320, 320, 3)
n_classes = 90
mode = 'inference'
l2_regularization = 0.0004
min_scale = 0.2 #None
max_scale = 0.95 #None
scales =  [0.1, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0]
aspect_ratios_global = None #[1.0, 2.0, 0.5, 3.0, 1.0 / 3.0] # None
aspect_ratios_per_layer = [[1.01, 2.0, 0.5], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]]
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
confidence_thresh = 0.5 # 0.65
iou_threshold = 0.6
top_k = 100 #200
nms_max_output_size = 100 #400
return_predictor_sizes = False
model_type="small_extractor"
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

print(model.summary())
# FeatureExtractor
print(model.layers[1].summary())
print(model.layers[1].layers[3].summary())
# # MobileNetv3
print(model.layers[1].layers[3].layers[0].summary())


# 2: Load the trained weights into the model.
initial_weights = [layer.get_weights() for layer in model.layers]

# weights_path = './ckpt/tf_convert_mbv3ssdlite_large_0528.h5'
# weights_path = './ckpt/tf_convert_mbv3ssdlite_small_0527.h5'

def dfsmodel(prefix,layers,f):
    # for key,value in f.items():
        # print("cur f:",key,value)

    for layer in layers:
        # print(prefix + '/' +layer.name)
        if hasattr(layer, 'layers'):
            layer = dfsmodel(prefix+'/'+layer.name,layer.layers,f[layer.name])
        else:
            print(prefix + '/' +layer.name)
            if(len(layer.weights)>0):
                print(layer.name,f[layer.name])

                layer.set_weights(f[layer.name])
                # global weights_num
                # weights_num+=1
                print("loaded {} weights".format(prefix + '/' +layer.name))
    return layers

# load weights 
# h5fn = h5py.File(weights_path,'r')
# dfsmodel('',model.layers[1].layers,h5fn)

# model.load_weights("./ckpt/loaded_large_0528_weights.h5",by_name=True)
model.load_weights("./ckpt/loaded_small_0527_weights.h5",by_name=True)


for layer, initial in zip(model.layers, initial_weights):
    weights = layer.get_weights()
    
    print(layer.name)
    print("len(weights)",len(weights))

    # print(initial)
    if weights and all(tf.nest.map_structure(np.array_equal, weights, initial)):
        print(f'first loaded contained no weights for layer {layer.name}!')

# h5 weights loaded successfully

# save weights
# model.save_weights("./ckpt/loaded_large_0528_weights.h5")
# model.save_weights("./ckpt/loaded_small_0527_weights.h5")

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './testfile/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  imgdata = np.array(image.getdata())
  if(len(imgdata.shape)==1):
      return imgdata.reshape((im_height, im_width, 1)).astype(np.uint8).repeat(3,axis=2)
  return imgdata.reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# We'll only load one image in this example.
img_dir = './imgs/test'

for imgfn in os.listdir(img_dir):
    img_path = os.path.join(img_dir, imgfn)
    print(img_path)
    orig_images = [] # Store the images here.
    input_images = [] # Store resized versions of the images here.

    orig_images.append(imread(img_path))
    
    # keras preprocess
    # img = image.load_img(img_path, target_size=(image_size[0], image_size[1]))
    # img = image.img_to_array(img)
    # input_images.append(img)

    image = Image.open(img_path)
    image = image.resize((image_size[0],image_size[1]),Image.ANTIALIAS)
    image_np = load_image_into_numpy_array(image)
    input_images.append(image_np)

    input_images = np.array(input_images,dtype=np.float32)
    print("origin image",input_images)
    # tf preprocess
    input_images = input_images*(2.0 / 255.0)-1.0
    print("image",input_images)

    # outputs = model.predict(input_images)
    y_pred = model.predict(input_images)
    
    print("predict end.")

    #confidence_threshold = 0.5
    confidence_threshold = confidence_thresh

    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print("y_pred_thresh[0]",y_pred_thresh[0])

    colors = plt.cm.hsv(np.linspace(0, 1, 91)).tolist()

    plt.figure(figsize=(20, 12))
    plt.imshow(orig_images[0])

    current_axis = plt.gca()

    # print("y_pred_thresh[0]",y_pred_thresh[0])

    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2] * orig_images[0].shape[1] / image_size[0]
        ymin = box[3] * orig_images[0].shape[0] / image_size[1]
        xmax = box[4] * orig_images[0].shape[1] / image_size[1]
        ymax = box[5] * orig_images[0].shape[0] / image_size[0]
        
        # print("xmin:{},ymin:{},xmax:{},ymax:{}".format(xmin,ymin,xmax,ymax))

        color = colors[int(box[0])]
        if(int(box[0]) not in category_index.keys()):
            continue
        clsname = category_index[int(box[0])]['name']
        label = '{}: {:.2f}'.format(clsname, box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
    #plt.show()
    plt.savefig("./imgs/detections_test/keras_{}".format(imgfn))
    plt.ion()
    plt.pause(2)
    plt.close()
