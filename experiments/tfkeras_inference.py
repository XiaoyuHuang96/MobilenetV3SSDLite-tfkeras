from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
from losses.keras_ssd_loss import SSDLoss
# from models.keras_mobilenet_v2_ssdlite import mobilenet_v2_ssd
from models.mbv3_ssdlite import mobilenet_v3_ssd
import os
import tensorflow as tf
# from ckpt2h5 import load_weights_from_tf_checkpoint,MyCallbacks
from tensorflow.keras.utils import plot_model


# model config
batch_size = 32
image_size = (300, 300, 3)
n_classes = 80
mode = 'inference'
l2_regularization = 0.0005
min_scale = 0.1 #None
max_scale = 0.95 #None
scales = [0.1, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0] #[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
aspect_ratios_global = None
aspect_ratios_per_layer = [[1.0, 2.0, 0.5], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5], [1.0, 2.0, 0.5]]
two_boxes_for_ar1 = True
steps = None
offsets = None 
clip_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
coords = 'centroids'
normalize_coords = True
subtract_mean = [123, 117, 104]
divide_by_stddev = 128
swap_channels = None
confidence_thresh = 0.05 #0.65
iou_threshold = 0.45
top_k = 200
nms_max_output_size = 400
return_predictor_sizes = False
model_type="large_extractor"
width_multiplier=1.0
model_name="FeatureExtractor_Small"#"Mobilenetv3_Large"
divisible_by=8
K.clear_session()
K.set_learning_phase(0)

model = mobilenet_v3_ssd(image_size, n_classes, mode, l2_regularization, min_scale, max_scale, scales,
                         aspect_ratios_global, aspect_ratios_per_layer, two_boxes_for_ar1, steps,
                         offsets, clip_boxes, variances, coords, normalize_coords, subtract_mean,
                         divide_by_stddev, swap_channels, confidence_thresh, iou_threshold, top_k,
                         nms_max_output_size, return_predictor_sizes,model_type,width_multiplier,model_name,divisible_by)

print(model.summary())


# 2: Load the trained weights into the model.
# weights_path = './trained_models/other_model.h5'
# weights_path = './pretrained_weights/ssd_mobilenet_v3_large_coco_2019_08_14/model.ckpt'
# h5_path = './trained_models/tf_ckpt_400000.h5'
# weights_path = './pretrained_weights/ssdlite_coco_loss-4.8205_val_loss-4.1873.h5'
weights_path = './logs/mbv3_ssdlite_coco_12_loss-8.1651_val_loss-7.7919.h5'
# model.load_weights(weights_path, by_name=False)

# model=tf.keras.models.load_model("./pretrained_weights/ssd_mobilenet_v3_large_coco_2019_08_14/")
# print("loaded",model)
# load_weights_from_tf_checkpoint(model, weights_path, background_label=False)
# model.save_weights(h5_path)
# weights_path = './pretrained_weights/ssd_mobilenet_v3_large_coco_2019_08_14/model.h5'
model.load_weights(weights_path, by_name=True)


# We'll only load one image in this example.
img_dir = './imgs/cocotrain'
for img in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img)
    orig_images = [] # Store the images here.
    input_images = [] # Store resized versions of the images here.

    orig_images.append(imread(img_path))
    img = image.load_img(img_path, target_size=(image_size[0], image_size[1]))
    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images)

    y_pred = model.predict(input_images)

    #confidence_threshold = 0.5
    confidence_threshold = confidence_thresh

    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_thresh[0])

    colors = plt.cm.hsv(np.linspace(0, 1, 81)).tolist()

    plt.figure(figsize=(20, 12))
    plt.imshow(orig_images[0])

    current_axis = plt.gca()

    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2] * orig_images[0].shape[1] / image_size[0]
        ymin = box[3] * orig_images[0].shape[0] / image_size[1]
        xmax = box[4] * orig_images[0].shape[1] / image_size[1]
        ymax = box[5] * orig_images[0].shape[0] / image_size[0]
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
    #plt.show()
    plt.ion()
    plt.pause(2)
    plt.close()
