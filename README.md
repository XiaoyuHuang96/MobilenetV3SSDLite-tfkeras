# MobilenetV3SSDLite-tfkeras
tensorflow keras implement of mobilenet v3 ssdlite, same structure as tensorflow model.
some utils for converting ckpt model to keras model,and keras model to pb model.

##  Environments
+ python 3.6
+ tensorflow 1.14
+ cuda 10
+ cudnn 7.6.5
+ pycocotools
+ [tensorflow object detection api](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

## Introductions
### overall
+ tf ckpt model and keras model have some difference in preprocess phase, model details and anchor settings.
+ In this repo, we don't use variance(anchor setting parameter).

### model structure
+ this model is build by tf.keras.models
+ model files in ./models 

### convert tf ckpt to keras models
+ origin ckpt file's inference file in ./experiments/tfckpt_inference.py
+ this repo can convert tf ckpt models to tfkeras h5 models, main file in ./model_transformation_utils/load_weights_tf2keras.py 
+ other files in ./model_transformation_utils/ are my other tries , can be understanded by their names.
+ keras to pb is work but keras to tflite is not work, because some ops in decodelayer are not supported yet in tflite.

### convert model's inference and test
+ code for eval converted model's performance
+ at first, use dfsmodel function(line 121) to load ckpt to keras model's weights, otherwise it will not be loaded successfully.
+ After first loading phase, you can use save_weights function to save now weights for this keras models. 
+ Then you can use load_weights to load now weights for inference/test keras models.

+ infer several images by ./experiments/tfckpt2keras_inference.py
+ eval performance by ./experiments/tfckpt2keras_test.py

### keras model's train,inference and test
+ you can also use this repo to train your own keras models.
+ this repo can also be trained by ./experiments/tfkeras_train.py
+ this repo can be inferenced by ./experiments/tfkeras_inference.py
+ this repo can be evaled by ./experiments/tfkeras_test.py


## reference
[1] https://github.com/tensorflow/models/tree/master/research/object_detection

[2] https://github.com/markshih91/mobilenet_v2_ssdlite_keras

[3] https://github.com/Bisonai/mobilenetv3-tensorflow