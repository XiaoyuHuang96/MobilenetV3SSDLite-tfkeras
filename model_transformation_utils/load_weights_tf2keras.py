import os
import numpy as np
import tensorflow as tf
import h5py
import argparse



ckpt2kerasdict_small={
    # backbone
    "FeatureExtractor/MobilenetV3/Conv": "FeatureExtractor/MobilenetV3/FirstLayer",
    "FeatureExtractor/MobilenetV3/Conv_1": "FeatureExtractor/MobilenetV3/LastStage/Conv_1",
    # box 0
    "BoxPredictor_0/BoxEncodingPredictor/weights":"BoxPredictor_0_BoxEncodingPredictor/BoxEncodingPredictor0_conv1/kernel:0",
    "BoxPredictor_0/BoxEncodingPredictor/biases":"BoxPredictor_0_BoxEncodingPredictor/BoxEncodingPredictor0_conv1/biases:0",
    "BoxPredictor_0/BoxEncodingPredictor_depthwise/BatchNorm":"BoxPredictor_0_BoxEncodingPredictor/BoxEncodingPredictor0_depthwise/BatchNorm",
    "BoxPredictor_0/BoxEncodingPredictor_depthwise/depthwise_weights":"BoxPredictor_0_BoxEncodingPredictor/BoxEncodingPredictor0_depthwise/depthwise_kernel:0",
    "BoxPredictor_0/ClassPredictor/weights":"BoxPredictor_0_ClassPredictor/ClassPredictor0_conv1/kernel:0",
    "BoxPredictor_0/ClassPredictor/biases":"BoxPredictor_0_ClassPredictor/ClassPredictor0_conv1/biases:0",
    "BoxPredictor_0/ClassPredictor_depthwise/BatchNorm":"BoxPredictor_0_ClassPredictor/ClassPredictor0_depthwise/BatchNorm",
    "BoxPredictor_0/ClassPredictor_depthwise/depthwise_weights":"BoxPredictor_0_ClassPredictor/ClassPredictor0_depthwise/depthwise_kernel:0",
    # box 1
    "BoxPredictor_1/BoxEncodingPredictor/weights":"BoxPredictor_1_BoxEncodingPredictor/BoxEncodingPredictor1_conv1/kernel:0",
    "BoxPredictor_1/BoxEncodingPredictor/biases":"BoxPredictor_1_BoxEncodingPredictor/BoxEncodingPredictor1_conv1/biases:0",
    "BoxPredictor_1/BoxEncodingPredictor_depthwise/BatchNorm":"BoxPredictor_1_BoxEncodingPredictor/BoxEncodingPredictor1_depthwise/BatchNorm",
    "BoxPredictor_1/BoxEncodingPredictor_depthwise/depthwise_weights":"BoxPredictor_1_BoxEncodingPredictor/BoxEncodingPredictor1_depthwise/depthwise_kernel:0",
    "BoxPredictor_1/ClassPredictor/weights":"BoxPredictor_1_ClassPredictor/ClassPredictor1_conv1/kernel:0",
    "BoxPredictor_1/ClassPredictor/biases":"BoxPredictor_1_ClassPredictor/ClassPredictor1_conv1/biases:0",
    "BoxPredictor_1/ClassPredictor_depthwise/BatchNorm":"BoxPredictor_1_ClassPredictor/ClassPredictor1_depthwise/BatchNorm",
    "BoxPredictor_1/ClassPredictor_depthwise/depthwise_weights":"BoxPredictor_1_ClassPredictor/ClassPredictor1_depthwise/depthwise_kernel:0",
    # box 2
    "BoxPredictor_2/BoxEncodingPredictor/weights":"BoxPredictor_2_BoxEncodingPredictor/BoxEncodingPredictor2_conv1/kernel:0",
    "BoxPredictor_2/BoxEncodingPredictor/biases":"BoxPredictor_2_BoxEncodingPredictor/BoxEncodingPredictor2_conv1/biases:0",
    "BoxPredictor_2/BoxEncodingPredictor_depthwise/BatchNorm":"BoxPredictor_2_BoxEncodingPredictor/BoxEncodingPredictor2_depthwise/BatchNorm",
    "BoxPredictor_2/BoxEncodingPredictor_depthwise/depthwise_weights":"BoxPredictor_2_BoxEncodingPredictor/BoxEncodingPredictor2_depthwise/depthwise_kernel:0",
    "BoxPredictor_2/ClassPredictor/weights":"BoxPredictor_2_ClassPredictor/ClassPredictor2_conv1/kernel:0",
    "BoxPredictor_2/ClassPredictor/biases":"BoxPredictor_2_ClassPredictor/ClassPredictor2_conv1/biases:0",
    "BoxPredictor_2/ClassPredictor_depthwise/BatchNorm":"BoxPredictor_2_ClassPredictor/ClassPredictor2_depthwise/BatchNorm",
    "BoxPredictor_2/ClassPredictor_depthwise/depthwise_weights":"BoxPredictor_2_ClassPredictor/ClassPredictor2_depthwise/depthwise_kernel:0",
    # box 3
    "BoxPredictor_3/BoxEncodingPredictor/weights":"BoxPredictor_3_BoxEncodingPredictor/BoxEncodingPredictor3_conv1/kernel:0",
    "BoxPredictor_3/BoxEncodingPredictor/biases":"BoxPredictor_3_BoxEncodingPredictor/BoxEncodingPredictor3_conv1/biases:0",
    "BoxPredictor_3/BoxEncodingPredictor_depthwise/BatchNorm":"BoxPredictor_3_BoxEncodingPredictor/BoxEncodingPredictor3_depthwise/BatchNorm",
    "BoxPredictor_3/BoxEncodingPredictor_depthwise/depthwise_weights":"BoxPredictor_3_BoxEncodingPredictor/BoxEncodingPredictor3_depthwise/depthwise_kernel:0",
    "BoxPredictor_3/ClassPredictor/weights":"BoxPredictor_3_ClassPredictor/ClassPredictor3_conv1/kernel:0",
    "BoxPredictor_3/ClassPredictor/biases":"BoxPredictor_3_ClassPredictor/ClassPredictor3_conv1/biases:0",
    "BoxPredictor_3/ClassPredictor_depthwise/BatchNorm":"BoxPredictor_3_ClassPredictor/ClassPredictor3_depthwise/BatchNorm",
    "BoxPredictor_3/ClassPredictor_depthwise/depthwise_weights":"BoxPredictor_3_ClassPredictor/ClassPredictor3_depthwise/depthwise_kernel:0",
    # box 4
    "BoxPredictor_4/BoxEncodingPredictor/weights":"BoxPredictor_4_BoxEncodingPredictor/BoxEncodingPredictor4_conv1/kernel:0",
    "BoxPredictor_4/BoxEncodingPredictor/biases":"BoxPredictor_4_BoxEncodingPredictor/BoxEncodingPredictor4_conv1/biases:0",
    "BoxPredictor_4/BoxEncodingPredictor_depthwise/BatchNorm":"BoxPredictor_4_BoxEncodingPredictor/BoxEncodingPredictor4_depthwise/BatchNorm",
    "BoxPredictor_4/BoxEncodingPredictor_depthwise/depthwise_weights":"BoxPredictor_4_BoxEncodingPredictor/BoxEncodingPredictor4_depthwise/depthwise_kernel:0",
    "BoxPredictor_4/ClassPredictor/weights":"BoxPredictor_4_ClassPredictor/ClassPredictor4_conv1/kernel:0",
    "BoxPredictor_4/ClassPredictor/biases":"BoxPredictor_4_ClassPredictor/ClassPredictor4_conv1/biases:0",
    "BoxPredictor_4/ClassPredictor_depthwise/BatchNorm":"BoxPredictor_4_ClassPredictor/ClassPredictor4_depthwise/BatchNorm",
    "BoxPredictor_4/ClassPredictor_depthwise/depthwise_weights":"BoxPredictor_4_ClassPredictor/ClassPredictor4_depthwise/depthwise_kernel:0",
    # box 5
    "BoxPredictor_5/BoxEncodingPredictor/weights":"BoxPredictor_5_BoxEncodingPredictor/BoxEncodingPredictor5_conv1/kernel:0",
    "BoxPredictor_5/BoxEncodingPredictor/biases":"BoxPredictor_5_BoxEncodingPredictor/BoxEncodingPredictor5_conv1/biases:0",
    "BoxPredictor_5/BoxEncodingPredictor_depthwise/BatchNorm":"BoxPredictor_5_BoxEncodingPredictor/BoxEncodingPredictor5_depthwise/BatchNorm",
    "BoxPredictor_5/BoxEncodingPredictor_depthwise/depthwise_weights":"BoxPredictor_5_BoxEncodingPredictor/BoxEncodingPredictor5_depthwise/depthwise_kernel:0",
    "BoxPredictor_5/ClassPredictor/weights":"BoxPredictor_5_ClassPredictor/ClassPredictor5_conv1/kernel:0",
    "BoxPredictor_5/ClassPredictor/biases":"BoxPredictor_5_ClassPredictor/ClassPredictor5_conv1/biases:0",
    "BoxPredictor_5/ClassPredictor_depthwise/BatchNorm":"BoxPredictor_5_ClassPredictor/ClassPredictor5_depthwise/BatchNorm",
    "BoxPredictor_5/ClassPredictor_depthwise/depthwise_weights":"BoxPredictor_5_ClassPredictor/ClassPredictor5_depthwise/depthwise_kernel:0",

    # downsample 2
    "FeatureExtractor/MobilenetV3/layer_13_1_Conv2d_2_1x1_256/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_2/layer_13_1_Conv2d_2_1x1_256/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_13_1_Conv2d_2_1x1_256/weights":"FeatureExtractor/ssdDownSampleBlock_2/layer_13_1_Conv2d_2_1x1_256/kernel:0",
    "FeatureExtractor/MobilenetV3/layer_13_2_Conv2d_2_3x3_s2_512_depthwise/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_2/layer_13_2_Conv2d_2_3x3_s2_512_depthwise/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_13_2_Conv2d_2_3x3_s2_512_depthwise/depthwise_weights":"FeatureExtractor/ssdDownSampleBlock_2/layer_13_2_Conv2d_2_3x3_s2_512_depthwise/depthwise_kernel:0",
    "FeatureExtractor/MobilenetV3/layer_13_2_Conv2d_2_3x3_s2_512/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_2/layer_13_2_Conv2d_2_3x3_s2_512/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_13_2_Conv2d_2_3x3_s2_512/weights":"FeatureExtractor/ssdDownSampleBlock_2/layer_13_2_Conv2d_2_3x3_s2_512/kernel:0",
    # downsample 3
    "FeatureExtractor/MobilenetV3/layer_13_1_Conv2d_3_1x1_128/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_3/layer_13_1_Conv2d_3_1x1_128/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_13_1_Conv2d_3_1x1_128/weights":"FeatureExtractor/ssdDownSampleBlock_3/layer_13_1_Conv2d_3_1x1_128/kernel:0",
    "FeatureExtractor/MobilenetV3/layer_13_2_Conv2d_3_3x3_s2_256_depthwise/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_3/layer_13_2_Conv2d_3_3x3_s2_256_depthwise/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_13_2_Conv2d_3_3x3_s2_256_depthwise/depthwise_weights":"FeatureExtractor/ssdDownSampleBlock_3/layer_13_2_Conv2d_3_3x3_s2_256_depthwise/depthwise_kernel:0",
    "FeatureExtractor/MobilenetV3/layer_13_2_Conv2d_3_3x3_s2_256/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_3/layer_13_2_Conv2d_3_3x3_s2_256/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_13_2_Conv2d_3_3x3_s2_256/weights":"FeatureExtractor/ssdDownSampleBlock_3/layer_13_2_Conv2d_3_3x3_s2_256/kernel:0",
    # downsample 4
    "FeatureExtractor/MobilenetV3/layer_13_1_Conv2d_4_1x1_128/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_4/layer_13_1_Conv2d_4_1x1_128/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_13_1_Conv2d_4_1x1_128/weights":"FeatureExtractor/ssdDownSampleBlock_4/layer_13_1_Conv2d_4_1x1_128/kernel:0",
    "FeatureExtractor/MobilenetV3/layer_13_2_Conv2d_4_3x3_s2_256/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_4/layer_13_2_Conv2d_4_3x3_s2_256/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_13_2_Conv2d_4_3x3_s2_256/weights":"FeatureExtractor/ssdDownSampleBlock_4/layer_13_2_Conv2d_4_3x3_s2_256/kernel:0",
    "FeatureExtractor/MobilenetV3/layer_13_2_Conv2d_4_3x3_s2_256_depthwise/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_4/layer_13_2_Conv2d_4_3x3_s2_256_depthwise/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_13_2_Conv2d_4_3x3_s2_256_depthwise/depthwise_weights":"FeatureExtractor/ssdDownSampleBlock_4/layer_13_2_Conv2d_4_3x3_s2_256_depthwise/depthwise_kernel:0",
    # downsample 5
    "FeatureExtractor/MobilenetV3/layer_13_1_Conv2d_5_1x1_64/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_5/layer_13_1_Conv2d_5_1x1_64/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_13_1_Conv2d_5_1x1_64/weights":"FeatureExtractor/ssdDownSampleBlock_5/layer_13_1_Conv2d_5_1x1_64/kernel:0",
    "FeatureExtractor/MobilenetV3/layer_13_2_Conv2d_5_3x3_s2_128/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_5/layer_13_2_Conv2d_5_3x3_s2_128/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_13_2_Conv2d_5_3x3_s2_128/weights":"FeatureExtractor/ssdDownSampleBlock_5/layer_13_2_Conv2d_5_3x3_s2_128/kernel:0",
    "FeatureExtractor/MobilenetV3/layer_13_2_Conv2d_5_3x3_s2_128_depthwise/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_5/layer_13_2_Conv2d_5_3x3_s2_128_depthwise/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_13_2_Conv2d_5_3x3_s2_128_depthwise/depthwise_weights":"FeatureExtractor/ssdDownSampleBlock_5/layer_13_2_Conv2d_5_3x3_s2_128_depthwise/depthwise_kernel:0",
}

ckpt2kerasdict_large={
    # backbone
    "FeatureExtractor/MobilenetV3/Conv": "FeatureExtractor/MobilenetV3/FirstLayer",
    "FeatureExtractor/MobilenetV3/Conv_1": "FeatureExtractor/MobilenetV3/LastStage/Conv_1",
    # box 0
    "BoxPredictor_0/BoxEncodingPredictor/weights":"BoxPredictor_0_BoxEncodingPredictor/BoxEncodingPredictor0_conv1/kernel:0",
    "BoxPredictor_0/BoxEncodingPredictor/biases":"BoxPredictor_0_BoxEncodingPredictor/BoxEncodingPredictor0_conv1/biases:0",
    "BoxPredictor_0/BoxEncodingPredictor_depthwise/BatchNorm":"BoxPredictor_0_BoxEncodingPredictor/BoxEncodingPredictor0_depthwise/BatchNorm",
    "BoxPredictor_0/BoxEncodingPredictor_depthwise/depthwise_weights":"BoxPredictor_0_BoxEncodingPredictor/BoxEncodingPredictor0_depthwise/depthwise_kernel:0",
    "BoxPredictor_0/ClassPredictor/weights":"BoxPredictor_0_ClassPredictor/ClassPredictor0_conv1/kernel:0",
    "BoxPredictor_0/ClassPredictor/biases":"BoxPredictor_0_ClassPredictor/ClassPredictor0_conv1/biases:0",
    "BoxPredictor_0/ClassPredictor_depthwise/BatchNorm":"BoxPredictor_0_ClassPredictor/ClassPredictor0_depthwise/BatchNorm",
    "BoxPredictor_0/ClassPredictor_depthwise/depthwise_weights":"BoxPredictor_0_ClassPredictor/ClassPredictor0_depthwise/depthwise_kernel:0",
    # box 1
    "BoxPredictor_1/BoxEncodingPredictor/weights":"BoxPredictor_1_BoxEncodingPredictor/BoxEncodingPredictor1_conv1/kernel:0",
    "BoxPredictor_1/BoxEncodingPredictor/biases":"BoxPredictor_1_BoxEncodingPredictor/BoxEncodingPredictor1_conv1/biases:0",
    "BoxPredictor_1/BoxEncodingPredictor_depthwise/BatchNorm":"BoxPredictor_1_BoxEncodingPredictor/BoxEncodingPredictor1_depthwise/BatchNorm",
    "BoxPredictor_1/BoxEncodingPredictor_depthwise/depthwise_weights":"BoxPredictor_1_BoxEncodingPredictor/BoxEncodingPredictor1_depthwise/depthwise_kernel:0",
    "BoxPredictor_1/ClassPredictor/weights":"BoxPredictor_1_ClassPredictor/ClassPredictor1_conv1/kernel:0",
    "BoxPredictor_1/ClassPredictor/biases":"BoxPredictor_1_ClassPredictor/ClassPredictor1_conv1/biases:0",
    "BoxPredictor_1/ClassPredictor_depthwise/BatchNorm":"BoxPredictor_1_ClassPredictor/ClassPredictor1_depthwise/BatchNorm",
    "BoxPredictor_1/ClassPredictor_depthwise/depthwise_weights":"BoxPredictor_1_ClassPredictor/ClassPredictor1_depthwise/depthwise_kernel:0",
    # box 2
    "BoxPredictor_2/BoxEncodingPredictor/weights":"BoxPredictor_2_BoxEncodingPredictor/BoxEncodingPredictor2_conv1/kernel:0",
    "BoxPredictor_2/BoxEncodingPredictor/biases":"BoxPredictor_2_BoxEncodingPredictor/BoxEncodingPredictor2_conv1/biases:0",
    "BoxPredictor_2/BoxEncodingPredictor_depthwise/BatchNorm":"BoxPredictor_2_BoxEncodingPredictor/BoxEncodingPredictor2_depthwise/BatchNorm",
    "BoxPredictor_2/BoxEncodingPredictor_depthwise/depthwise_weights":"BoxPredictor_2_BoxEncodingPredictor/BoxEncodingPredictor2_depthwise/depthwise_kernel:0",
    "BoxPredictor_2/ClassPredictor/weights":"BoxPredictor_2_ClassPredictor/ClassPredictor2_conv1/kernel:0",
    "BoxPredictor_2/ClassPredictor/biases":"BoxPredictor_2_ClassPredictor/ClassPredictor2_conv1/biases:0",
    "BoxPredictor_2/ClassPredictor_depthwise/BatchNorm":"BoxPredictor_2_ClassPredictor/ClassPredictor2_depthwise/BatchNorm",
    "BoxPredictor_2/ClassPredictor_depthwise/depthwise_weights":"BoxPredictor_2_ClassPredictor/ClassPredictor2_depthwise/depthwise_kernel:0",
    # box 3
    "BoxPredictor_3/BoxEncodingPredictor/weights":"BoxPredictor_3_BoxEncodingPredictor/BoxEncodingPredictor3_conv1/kernel:0",
    "BoxPredictor_3/BoxEncodingPredictor/biases":"BoxPredictor_3_BoxEncodingPredictor/BoxEncodingPredictor3_conv1/biases:0",
    "BoxPredictor_3/BoxEncodingPredictor_depthwise/BatchNorm":"BoxPredictor_3_BoxEncodingPredictor/BoxEncodingPredictor3_depthwise/BatchNorm",
    "BoxPredictor_3/BoxEncodingPredictor_depthwise/depthwise_weights":"BoxPredictor_3_BoxEncodingPredictor/BoxEncodingPredictor3_depthwise/depthwise_kernel:0",
    "BoxPredictor_3/ClassPredictor/weights":"BoxPredictor_3_ClassPredictor/ClassPredictor3_conv1/kernel:0",
    "BoxPredictor_3/ClassPredictor/biases":"BoxPredictor_3_ClassPredictor/ClassPredictor3_conv1/biases:0",
    "BoxPredictor_3/ClassPredictor_depthwise/BatchNorm":"BoxPredictor_3_ClassPredictor/ClassPredictor3_depthwise/BatchNorm",
    "BoxPredictor_3/ClassPredictor_depthwise/depthwise_weights":"BoxPredictor_3_ClassPredictor/ClassPredictor3_depthwise/depthwise_kernel:0",
    # box 4
    "BoxPredictor_4/BoxEncodingPredictor/weights":"BoxPredictor_4_BoxEncodingPredictor/BoxEncodingPredictor4_conv1/kernel:0",
    "BoxPredictor_4/BoxEncodingPredictor/biases":"BoxPredictor_4_BoxEncodingPredictor/BoxEncodingPredictor4_conv1/biases:0",
    "BoxPredictor_4/BoxEncodingPredictor_depthwise/BatchNorm":"BoxPredictor_4_BoxEncodingPredictor/BoxEncodingPredictor4_depthwise/BatchNorm",
    "BoxPredictor_4/BoxEncodingPredictor_depthwise/depthwise_weights":"BoxPredictor_4_BoxEncodingPredictor/BoxEncodingPredictor4_depthwise/depthwise_kernel:0",
    "BoxPredictor_4/ClassPredictor/weights":"BoxPredictor_4_ClassPredictor/ClassPredictor4_conv1/kernel:0",
    "BoxPredictor_4/ClassPredictor/biases":"BoxPredictor_4_ClassPredictor/ClassPredictor4_conv1/biases:0",
    "BoxPredictor_4/ClassPredictor_depthwise/BatchNorm":"BoxPredictor_4_ClassPredictor/ClassPredictor4_depthwise/BatchNorm",
    "BoxPredictor_4/ClassPredictor_depthwise/depthwise_weights":"BoxPredictor_4_ClassPredictor/ClassPredictor4_depthwise/depthwise_kernel:0",
    # box 5
    "BoxPredictor_5/BoxEncodingPredictor/weights":"BoxPredictor_5_BoxEncodingPredictor/BoxEncodingPredictor5_conv1/kernel:0",
    "BoxPredictor_5/BoxEncodingPredictor/biases":"BoxPredictor_5_BoxEncodingPredictor/BoxEncodingPredictor5_conv1/biases:0",
    "BoxPredictor_5/BoxEncodingPredictor_depthwise/BatchNorm":"BoxPredictor_5_BoxEncodingPredictor/BoxEncodingPredictor5_depthwise/BatchNorm",
    "BoxPredictor_5/BoxEncodingPredictor_depthwise/depthwise_weights":"BoxPredictor_5_BoxEncodingPredictor/BoxEncodingPredictor5_depthwise/depthwise_kernel:0",
    "BoxPredictor_5/ClassPredictor/weights":"BoxPredictor_5_ClassPredictor/ClassPredictor5_conv1/kernel:0",
    "BoxPredictor_5/ClassPredictor/biases":"BoxPredictor_5_ClassPredictor/ClassPredictor5_conv1/biases:0",
    "BoxPredictor_5/ClassPredictor_depthwise/BatchNorm":"BoxPredictor_5_ClassPredictor/ClassPredictor5_depthwise/BatchNorm",
    "BoxPredictor_5/ClassPredictor_depthwise/depthwise_weights":"BoxPredictor_5_ClassPredictor/ClassPredictor5_depthwise/depthwise_kernel:0",

    # downsample 2
    "FeatureExtractor/MobilenetV3/layer_17_1_Conv2d_2_1x1_256/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_2/layer_17_1_Conv2d_2_1x1_256/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_17_1_Conv2d_2_1x1_256/weights":"FeatureExtractor/ssdDownSampleBlock_2/layer_17_1_Conv2d_2_1x1_256/kernel:0",
    "FeatureExtractor/MobilenetV3/layer_17_2_Conv2d_2_3x3_s2_512_depthwise/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_2/layer_17_2_Conv2d_2_3x3_s2_512_depthwise/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_17_2_Conv2d_2_3x3_s2_512_depthwise/depthwise_weights":"FeatureExtractor/ssdDownSampleBlock_2/layer_17_2_Conv2d_2_3x3_s2_512_depthwise/depthwise_kernel:0",
    "FeatureExtractor/MobilenetV3/layer_17_2_Conv2d_2_3x3_s2_512/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_2/layer_17_2_Conv2d_2_3x3_s2_512/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_17_2_Conv2d_2_3x3_s2_512/weights":"FeatureExtractor/ssdDownSampleBlock_2/layer_17_2_Conv2d_2_3x3_s2_512/kernel:0",
    # downsample 3
    "FeatureExtractor/MobilenetV3/layer_17_1_Conv2d_3_1x1_128/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_3/layer_17_1_Conv2d_3_1x1_128/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_17_1_Conv2d_3_1x1_128/weights":"FeatureExtractor/ssdDownSampleBlock_3/layer_17_1_Conv2d_3_1x1_128/kernel:0",
    "FeatureExtractor/MobilenetV3/layer_17_2_Conv2d_3_3x3_s2_256_depthwise/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_3/layer_17_2_Conv2d_3_3x3_s2_256_depthwise/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_17_2_Conv2d_3_3x3_s2_256_depthwise/depthwise_weights":"FeatureExtractor/ssdDownSampleBlock_3/layer_17_2_Conv2d_3_3x3_s2_256_depthwise/depthwise_kernel:0",
    "FeatureExtractor/MobilenetV3/layer_17_2_Conv2d_3_3x3_s2_256/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_3/layer_17_2_Conv2d_3_3x3_s2_256/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_17_2_Conv2d_3_3x3_s2_256/weights":"FeatureExtractor/ssdDownSampleBlock_3/layer_17_2_Conv2d_3_3x3_s2_256/kernel:0",
    # downsample 4
    "FeatureExtractor/MobilenetV3/layer_17_1_Conv2d_4_1x1_128/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_4/layer_17_1_Conv2d_4_1x1_128/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_17_1_Conv2d_4_1x1_128/weights":"FeatureExtractor/ssdDownSampleBlock_4/layer_17_1_Conv2d_4_1x1_128/kernel:0",
    "FeatureExtractor/MobilenetV3/layer_17_2_Conv2d_4_3x3_s2_256/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_4/layer_17_2_Conv2d_4_3x3_s2_256/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_17_2_Conv2d_4_3x3_s2_256/weights":"FeatureExtractor/ssdDownSampleBlock_4/layer_17_2_Conv2d_4_3x3_s2_256/kernel:0",
    "FeatureExtractor/MobilenetV3/layer_17_2_Conv2d_4_3x3_s2_256_depthwise/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_4/layer_17_2_Conv2d_4_3x3_s2_256_depthwise/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_17_2_Conv2d_4_3x3_s2_256_depthwise/depthwise_weights":"FeatureExtractor/ssdDownSampleBlock_4/layer_17_2_Conv2d_4_3x3_s2_256_depthwise/depthwise_kernel:0",
    # downsample 5
    "FeatureExtractor/MobilenetV3/layer_17_1_Conv2d_5_1x1_64/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_5/layer_17_1_Conv2d_5_1x1_64/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_17_1_Conv2d_5_1x1_64/weights":"FeatureExtractor/ssdDownSampleBlock_5/layer_17_1_Conv2d_5_1x1_64/kernel:0",
    "FeatureExtractor/MobilenetV3/layer_17_2_Conv2d_5_3x3_s2_128/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_5/layer_17_2_Conv2d_5_3x3_s2_128/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_17_2_Conv2d_5_3x3_s2_128/weights":"FeatureExtractor/ssdDownSampleBlock_5/layer_17_2_Conv2d_5_3x3_s2_128/kernel:0",
    "FeatureExtractor/MobilenetV3/layer_17_2_Conv2d_5_3x3_s2_128_depthwise/BatchNorm":"FeatureExtractor/ssdDownSampleBlock_5/layer_17_2_Conv2d_5_3x3_s2_128_depthwise/BatchNorm",
    "FeatureExtractor/MobilenetV3/layer_17_2_Conv2d_5_3x3_s2_128_depthwise/depthwise_weights":"FeatureExtractor/ssdDownSampleBlock_5/layer_17_2_Conv2d_5_3x3_s2_128_depthwise/depthwise_kernel:0",
}



def isKeyValid(key):
    validset = {"biases","weights","depthwise_weights","beta","gamma","moving_mean","moving_variance","Momentum"}
    keysplits = key.split(r'/')
    if keysplits[-1] in validset:
        return True
    return False

def setlayersweights(f,prefix,weightname,weights):
    splits_prefix = prefix.split(r'/')
    tmpf = f
    for i in range(len(splits_prefix)):
        if(splits_prefix[i] not in tmpf.keys()):
            tmpf.create_group(splits_prefix[i])
            print("create ",splits_prefix[i])
        tmpf = tmpf[splits_prefix[i]]
    tmpf[weightname] = weights
    print("set",prefix,weightname)
    return f

def load_weights_tf2keras(ckpt,h5fn):
    reader = tf.compat.v1.train.NewCheckpointReader(ckpt)
    f = h5py.File(h5fn,'w')
    t_g = None
    
    used = set()

    for key in sorted(reader.get_variable_to_shape_map()):
        # print(key,reader.get_tensor(key).shape)
        print("all keys",key,reader.get_tensor(key).shape)
        # 权重名称根据自己网络名称进行修改
        if(isKeyValid(key)):
            keySplits = key.split(r'/')
            print(key,len(keySplits))
            if(keySplits[-1]=='Momentum'and keySplits[3]!='squeeze_excite'):
                continue
            if(len(keySplits)==3):
                # box's weights,depthwise_weights
                tmpconv =  keySplits[0]+'/'+keySplits[1]+'/'+keySplits[2]
                if(tmpconv in ckpt2kerasdict):
                    print("find {} in ckpt2kerasdict".format(tmpconv))
                    tmppath = ckpt2kerasdict[tmpconv]
                    print("map to",tmppath)

                    tmpsplits = tmppath.split(r'/')
                    respath = tmpsplits[0]
                    for i in range(1,len(tmpsplits)-1):
                        respath += '/'+tmpsplits[i]
                    weightsname = tmpsplits[-1]
                    # print("respath",respath,"weightsname",weightsname)
                    fullname = respath+'/'+weightsname
                    if(fullname not in used):
                        used.add(fullname)
                        f = setlayersweights(f,respath,weightsname,reader.get_tensor(key))
                        # if(respath not in f.keys()):
                        #     f.create_group(respath)
                        # f[respath][weightsname] = reader.get_tensor(key)
                        
            elif(len(keySplits)>3):
                # first/last stage
                tmpconv =  keySplits[0]+'/'+keySplits[1]+'/'+keySplits[2]
                if(tmpconv in ckpt2kerasdict):
                    respath = ckpt2kerasdict[tmpconv]
                    # first/last stage
                    if(len(keySplits)==4 and keySplits[3]=='weights'):
                        respath+='/'+keySplits[2]
                        weightsname = 'kernel:0'
                    else:
                        # bn
                        for i in range(3,len(keySplits)-1):
                            respath += '/' + keySplits[i]
                        weightsname = keySplits[-1]+':0'
                    # print("respath",respath,"weightsname",weightsname)
                    fullname = respath+'/'+weightsname
                    if(fullname not in used):
                        used.add(fullname)
                        f = setlayersweights(f,respath,weightsname,reader.get_tensor(key))
                        # if(respath not in f.keys()):
                        #     f.create_group(respath)
                        # f[respath][weightsname] = reader.get_tensor(key)
                elif tmpconv+'/'+keySplits[3] in ckpt2kerasdict:
                    # downsample
                    respath = ckpt2kerasdict[tmpconv+'/'+keySplits[3]]
                    if(len(keySplits)>4):
                        for i in range(4,len(keySplits)-1):
                            respath += '/' + keySplits[i]
                        weightsname = keySplits[-1] + ':0'
                        
                    else:
                        weightsname = respath.split(r'/')[-1]
                        respath = respath[0:respath.rfind('/')]
                    fullname = respath+'/'+weightsname
                    if(fullname not in used):
                        used.add(fullname)
                        f = setlayersweights(f,respath,weightsname,reader.get_tensor(key))
                        # if(respath not in f.keys()):
                        #     f.create_group(respath)
                        # f[respath][weightsname] = reader.get_tensor(key)
                elif(keySplits[3]=='squeeze_excite' and len(keySplits)==6):
                    print("squeeze_excite",len(keySplits),key)
                    # /Momentum
                    respath = tmpconv +'/'+keySplits[3]+'/'+keySplits[4]+'/'+keySplits[4]
                    if(keySplits[5]=='weights'):
                        weightsname = 'kernel:0'
                        print(reader.get_tensor(key)[0][0][0][0])
                    else:
                        weightsname = 'bias:0'
                        print(reader.get_tensor(key)[0])
                    # print("respath",respath,"weightsname",weightsname)
                    fullname = respath+'/'+weightsname
                    if(fullname not in used):
                        used.add(fullname)
                        f = setlayersweights(f,respath,weightsname,reader.get_tensor(key))
                        # if(respath not in f.keys()):
                        #     f.create_group(respath)
                        # f[respath][weightsname] = reader.get_tensor(key)
                elif(keySplits[3]=='project' or keySplits[3]=='expand'):
                    respath = tmpconv +'/'+keySplits[3]
                    
                    if(len(keySplits)==5):
                        respath += '/'+keySplits[3]# +'kernel:0'
                        weightsname = 'kernel:0'
                    else:
                        for i in range(4,len(keySplits)-1):
                            respath += '/' + keySplits[i]
                        weightsname = keySplits[-1] + ':0'
                    # print("respath",respath,"weightsname",weightsname)
                    fullname = respath+'/'+weightsname
                    if(fullname not in used):
                        used.add(fullname)
                        f = setlayersweights(f,respath,weightsname,reader.get_tensor(key))
                        # if(respath not in f.keys()):
                        #     f.create_group(respath)
                        # f[respath][weightsname] = reader.get_tensor(key)
                elif(keySplits[3]=='depthwise'):
                    respath = tmpconv +'/'+keySplits[3]
                    
                    if(len(keySplits)==5):
                        respath += '/'+keySplits[3]# +'kernel:0'
                        weightsname = 'depthwise_kernel:0'
                    else:
                        for i in range(4,len(keySplits)-1):
                            respath += '/' + keySplits[i]
                        weightsname = keySplits[-1] + ':0'
                    # print("respath",respath,"weightsname",weightsname)
                    fullname = respath+'/'+weightsname
                    if(fullname not in used):
                        used.add(fullname)
                        f = setlayersweights(f,respath,weightsname,reader.get_tensor(key))
                        # if(respath not in f.keys()):
                        #     f.create_group(respath)
                        # f[respath][weightsname] = reader.get_tensor(key)
                
            # else:
                # f[key] = reader.get_tensor(key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts ckpt weights to hdf5")
    parser.add_argument("infile", type=str,
                        help="Path to the ckpt.")
    parser.add_argument("outfile", type=str, nargs='?', default='',
                        help="Output file (inferred if missing).")
    parser.add_argument("model_tpye", type=str, nargs='?', default='small',
                        help="model type(small or large).")
    args = parser.parse_args()
    if(args.model_tpye=="small"):
        ckpt2kerasdict = ckpt2kerasdict_small
    else:
        ckpt2kerasdict = ckpt2kerasdict_large
    load_weights_tf2keras(args.infile,args.outfile)