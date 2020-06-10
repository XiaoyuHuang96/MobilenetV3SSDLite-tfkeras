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
+ This model is build by tf.keras.models
+ model files in ./models 

### convert tf ckpt to keras models
+ Origin ckpt file's inference file in ./experiments/tfckpt_inference.py
+ This repo can convert tf ckpt models to tfkeras h5 models, main file in ./model_transformation_utils/load_weights_tf2keras.py 
+ Other files in ./model_transformation_utils/ are my other tries , can be understanded by their names.
+ keras to pb is work but keras to tflite is not work, because some ops in decodelayer are not supported yet in tflite.

### converted model's inference and test
+ Code for eval converted model's performance
+ At first, use dfsmodel function(line 104) to load ckpt to keras model's weights, otherwise it will not be loaded successfully.
+ After first loading phase, you can use save_weights function to save now weights for this keras models. 
+ Then you can use load_weights to load now weights for inference/test keras models.

+ Infer several images by ./tfckpt2keras_inference.py
+ Eval performance by ./tfckpt2keras_test.py

### keras model's train,inference and test
+ You can also use this repo to train your own keras models.
+ This repo can be trained by ./tfkeras_train.py
+ This repo can be inferenced by ./tfkeras_inference.py
+ This repo can be evaled by ./tfkeras_test.py


## reference
[1] https://github.com/tensorflow/models/tree/master/research/object_detection

[2] https://github.com/markshih91/mobilenet_v2_ssdlite_keras

[3] https://github.com/Bisonai/mobilenetv3-tensorflow