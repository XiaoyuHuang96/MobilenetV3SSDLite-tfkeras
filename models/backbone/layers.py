# Copyright 2019 Bisonai Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of paper Searching for MobileNetV3, https://arxiv.org/abs/1905.02244

Layers of MobileNetV3
"""
import tensorflow as tf
import tensorflow.keras.layers as tflayers
from models.backbone.utils import get_layer
import numpy as np
# import tensorflow.contrib.slim as slim

def printlayerweights(layer):
    print(layer.name)
    for weight in layer.weights:
        print("weight name",weight.name, weight.shape)

def _make_divisible(v, divisor, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v

class Identity(tflayers.Layer):
    def __init__(self):
        super().__init__(name="Identity")

    def call(self, input):
        return input


class ReLU6(tflayers.Layer):
    def __init__(self):
        super().__init__(name="ReLU6")
        self.relu6 = tf.keras.layers.ReLU(max_value=6.0, name="ReLU6")

    def call(self, input):
        return self.relu6(input)


# class HardSigmoid(tflayers.Layer):
#     def __init__(self):
#         super().__init__(name="HardSigmoid")
#         self.relu6 = ReLU6()

#     def call(self, input):
#         return self.relu6(input + 3.0) / 6.0

class _HardSwish(tflayers.Layer):
    def __init__(self, name="_HardSwish"):
        super().__init__(name=name)
        self.relu6 = ReLU6()

    def call(self, input):
        return self.relu6(input + 3.0) *(1/ 6.0)

class HardSwish(tflayers.Layer):
    def __init__(self, name="HardSwish"):
        super().__init__(name=name)
        self.hard_swish = _HardSwish()

    def call(self, input):
        return input * self.hard_swish(input)


class Squeeze(tflayers.Layer):
    """Squeeze the second and third dimensions of given tensor.
    (batch, 1, 1, channels) -> (batch, channels)
    """
    def __init__(self):
        super().__init__(name="Squeeze")

    def call(self, input):
        x = tf.keras.backend.squeeze(input, 1)
        x = tf.keras.backend.squeeze(x, 1)
        return x


class GlobalAveragePooling2D(tflayers.Layer):
    """Return tensor of output shape (batch_size, rows, cols, channels)
    where rows and cols are equal to 1. Output shape of
    `tf.keras.layer.GlobalAveragePooling2D` is (batch_size, channels),
    """
    def __init__(self):
        super().__init__(name="GlobalAveragePooling2D")

    def build(self, input_shape):
        pool_size = tuple(map(int, input_shape[1:3]))
        self.gap = tf.keras.layers.AveragePooling2D(
            pool_size=pool_size,
            name="AvgPool{}x{}".format(pool_size[0],pool_size[1]),
        )

        super().build(input_shape)

    def call(self, input):
        return self.gap(input)
    
    def get_config(self):
        # config = super().get_config().copy()
        # config.update({
            # 'gap':self.gap
        # })
        config = {
            'gap':self.gap
        }

        # return config
        base_config = super(GlobalAveragePooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BatchNormalization(tflayers.Layer):
    """Searching fo MobileNetV3: All our convolutional layers
    use batch-normalization layers with average decay of 0.99.
    """
    def __init__(
            self,
            momentum: float=0.9,
            name="BatchNorm",
    ):
        super().__init__(name=name)
        self.momentum = momentum
        # self.name = name
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=0.9, epsilon= 0.001,#1e-5,
            name="BatchNorm",trainable = False,fused=True
        )

    def call(self, input):
        return self.bn(input)

    def set_weights(self,weights):
        bn_dicts = ["gamma:0","beta:0","moving_mean:0","moving_variance:0"]
        norm_weights = []
        
        for key in bn_dicts:
            print("bn keys",key)
            norm_weights.append(weights["BatchNorm"][key])
        self.bn.set_weights(norm_weights)

    def get_config(self):
        # config = super().get_config().copy()
        # config.update({
        #     'bn':self.bn
        # })
        config = {
            'momentum':self.momentum,
            # 'name':self.name,
        }
        # return config
        base_config = super(BatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvNormAct(tflayers.Layer):
# class ConvNormAct(tf.keras.models.Model):
    def __init__(
            self,
            filters: int,
            kernel_size: int=3,
            stride: int=1,
            padding: int=0,
            norm_layer: str=None,
            act_layer: str="relu",
            use_bias: bool=True,
            l2_reg: float=1e-5,
            name: str="ConvNormAct",
            
    ):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.use_bias = use_bias
        self.l2_reg = l2_reg
        # self.name = name

        # if padding > 0:
        #     self.pad = tf.keras.layers.ZeroPadding2D(
        #         padding=padding,
        #         name=name + "/Padding{}x{}".format(padding,padding),
        #     )
        # else:
        #     self.pad = Identity()

        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            name=name ,#+ "/Conv{}x{}".format(kernel_size,kernel_size),
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            use_bias=use_bias,
            padding="same",
        )

        _available_normalization = {
            # "bn": BatchNormalization(),
            "bn":tf.keras.layers.BatchNormalization(
                    momentum=0.9, epsilon= 1e-3,
                    name="BatchNorm",trainable = False,fused=True,
                ),
            # "bn":tf.layers.batch_normalization(
                    # momentum=0.9, epsilon= 1e-5,
                    # name="BatchNorm",trainable = False,fused=True,training=False)
            }
        self.norm = get_layer(norm_layer, _available_normalization, Identity())

        _available_activation = {
            "relu": tf.keras.layers.ReLU(name="ReLU"),
            "relu6": ReLU6(),
            "hswish": HardSwish(),
            # "hsigmoid": HardSigmoid(),
            "softmax": tf.keras.layers.Softmax(name="Softmax"),
        }
        self.act = get_layer(act_layer, _available_activation, Identity())

    def call(self, input):
        # x = self.pad(input)
        x = input
        x = self.conv(x)
        # printlayerweights(self.conv)
        # print("after "+self.conv.name,x.shape)
        x = self.norm(x)
        # printlayerweights(self.norm)
        x = self.act(x)
        return x
    
    def set_weights(self,weights):
        conv_dicts = ["kernel:0","bias:0"]
        conv_weights = []

        # for key,value in weights.items():
        #     print(key,value)

        #     for k,v in value.items():
        #         print(k,v)

        for key in conv_dicts:
            # print("conv keys",key)
            # if("Conv" in weights.)
            if self.name in weights.keys():
                if key in weights[self.conv.name].keys():
                    conv_weights.append(weights[self.conv.name][key])
            else:
                if key in weights["Conv"].keys():
                    conv_weights.append(weights["Conv"][key])
        self.conv.set_weights(conv_weights)

        if self.norm_layer:
            bn_dicts = ["gamma:0","beta:0","moving_mean:0","moving_variance:0"]
            norm_weights = []
            
            for key in bn_dicts:
                # print("bn keys",key)
                norm_weights.append(weights["BatchNorm"][key])
            self.norm.set_weights(norm_weights)

    def get_config(self):
        # config = super().get_config().copy()
        # config.update({
        #     'pad':self.pad,
        #     'conv':self.conv,
        #     'norm':self.norm,
        #     'act':self.act,
        # })
        config = {
            'filters':self.filters,
            'kernel_size':self.kernel_size,
            'stride':self.stride,
            'padding':self.padding,
            'norm_layer':self.norm_layer,
            'act_layer':self.act_layer,
            'use_bias':self.use_bias,
            'l2_reg':self.l2_reg,
            # 'name':self.name
        }
        # return config
        base_config = super(ConvNormAct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Bneck(tflayers.Layer):
# class Bneck(tf.keras.models.Model):
    def __init__(
            self,
            out_channels: int,
            exp_channels: int,
            kernel_size: int,
            stride: int,
            use_se: bool,
            act_layer: str,
            l2_reg: float=1e-5,
            has_expand: bool=False,
            name: str="Bneck",
    ):
        super().__init__(name=name)

        self.out_channels = out_channels
        self.exp_channels = exp_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_se = use_se
        self.has_expand = has_expand
        self.act_layer = act_layer
        self.l2_reg = l2_reg
        # self.name = name

        # Expand
        self.expand = ConvNormAct(
            exp_channels,
            kernel_size=1,
            norm_layer="bn",
            act_layer=act_layer,
            use_bias=False,
            l2_reg=l2_reg,
            name="expand",
        )

        # Depthwise
        # dw_padding = (kernel_size - 1) // 2
        # self.pad = tf.keras.layers.ZeroPadding2D(
        #     padding=dw_padding,
        #     name="Depthwise/Padding{}x{}".format(dw_padding,dw_padding),
        # )
        self.depthwise = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=stride,
            name="depthwise",#DWConv{kernel_size}x{kernel_size}",
            depthwise_regularizer=tf.keras.regularizers.l2(l2_reg),
            use_bias=False,
            padding="same",
        )
        # self.bn = BatchNormalization(name=name + "/depthwise/BatchNorm")
        self.bn = tf.keras.layers.BatchNormalization(
                    momentum=0.9, epsilon= 1e-3,
                    name="depthwise/BatchNorm",trainable = False,fused=True
                )
        if self.use_se:
            self.se = SEBottleneck(
                l2_reg=l2_reg,
                reduction=4,
                name="squeeze_excite",
            )

        _available_activation = {
            "relu": tf.keras.layers.ReLU(name=name+"/ReLU"),
            "hswish": HardSwish(name=name+"/HardSwish"),
        }
        self.act = get_layer(act_layer, _available_activation, Identity())

        # Project
        self.project = ConvNormAct(
            out_channels,
            kernel_size=1,
            norm_layer="bn",
            act_layer=None,
            use_bias=False,
            l2_reg=l2_reg,
            name="project",
        )

    def build(self, input_shape):
        self.in_channels = int(input_shape[3])
        super().build(input_shape)

    def call(self, input):
        if(self.name == "expanded_conv"):
            x = input
        else:
            x = self.expand(input)
        # printlayerweights(self.expand)
        # print("after "+self.expand.name,x.shape)
        expand_temp = x 
        # x = self.pad(x)
        x = self.depthwise(x)
        # printlayerweights(self.depthwise)
        # print("after "+self.depthwise.name,x.shape)
        x = self.bn(x,training=False)
        # printlayerweights(self.bn)
        x = self.act(x)
        if self.use_se:
            x = self.se(x)
            # printlayerweights(self.se)
            # print("after "+self.se.name,x.shape)
        x = self.project(x)
        # printlayerweights(self.project)
        # print("after "+self.project.name,x.shape)

        if(self.has_expand):
            if self.stride == 1 and self.in_channels == self.out_channels:
                return input + x, expand_temp
            else:
                return x, expand_temp
        else:
            if self.stride == 1 and self.in_channels == self.out_channels:
                return input + x
            else:
                return x
    
    def set_weights(self,weights):
        if("expand" in weights.keys()):
            self.expand.set_weights(weights["expand"])
        if("depthwise" in weights.keys()):
            # print("h5 weights",weights["depthwise"]["depthwise"]["depthwise_kernel:0"])
            # print("depthwise weights",np.array(self.depthwise.get_weights()).shape)
            self.depthwise.set_weights(np.expand_dims(weights["depthwise"]["depthwise"]["depthwise_kernel:0"], axis=0))
        if("depthwise" in weights.keys()):
            bn_dicts = ["gamma:0","beta:0","moving_mean:0","moving_variance:0"]
            norm_weights = []
            
            for key in bn_dicts:
                # print("bn keys",key)
                norm_weights.append(weights["depthwise"]["BatchNorm"][key])
            self.bn.set_weights(norm_weights)
        if("squeeze_excite" in weights.keys()):
            self.se.set_weights(weights["squeeze_excite"])
        if("project" in weights.keys()):
            self.project.set_weights(weights["project"])

    def get_config(self):
        # config = super().get_config().copy()
        # if(self.name != "expanded_conv"):
        #     config.update({
        #         'expand':self.expand
        #     })

        # config.update({
        #     'pad':self.pad,
        #     'depthwise':self.depthwise,
        #     'bn':self.bn,
        #     'act':self.act,
        # })
        
        # if self.use_se:
        #     config.update({'se':self.se})
        
        # config.update({'project':self.project})
        config = {
            'out_channels':self.out_channels,
            'exp_channels':self.exp_channels,
            'kernel_size':self.kernel_size,
            'stride':self.stride,
            'use_se':self.use_se,
            'has_expand':self.has_expand,
            'act_layer':self.act_layer,
            'l2_reg':self.l2_reg,
            # 'name':self.name,
        }

        # return config
        base_config = super(Bneck, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SEBottleneck(tflayers.Layer):
# class SEBottleneck(tf.keras.models.Model):
    def __init__(
            self,
            reduction: int=4,
            l2_reg: float=0.01,
            name: str="squeeze_excite",#"SEBottleneck",
    ):
        super().__init__(name=name)

        self.reduction = reduction
        self.l2_reg = l2_reg
        # self.name = name

    def build(self, input_shape):
        input_channels = int(input_shape[3])
        mid_channels = _make_divisible(input_channels / self.reduction, divisor=8)
        print("midchannels",mid_channels)
        self.gap = GlobalAveragePooling2D()
        self.conv1 = ConvNormAct(
            mid_channels,
            kernel_size=1,
            norm_layer=None,
            act_layer="relu",
            l2_reg=self.l2_reg,
            use_bias=True,
            name="Conv",#"Squeeze",
        )
        self.conv2 = ConvNormAct(
            input_channels,
            kernel_size=1,
            norm_layer=None,
            act_layer=None,#"hswish",
            use_bias=True,
            l2_reg=self.l2_reg,
            name="Conv_1",#"Excite",
        )
        self.act = _HardSwish() #tf.keras.activations.hard_sigmoid
        super().build(input_shape)

    def call(self, input):
        x = self.gap(input)
        # x = input
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        return input * x

    def set_weights(self,weights):
        # for key,value in weights.items():
        #     print("1-{}:{}".format(key,value))

        #     if(hasattr(value,"items")):
        #         for k,v in value.items():
        #             print("2-{}:{}".format(k,v))

        #             if(hasattr(v,"items")):
        #                 for k2,v2 in v.items():
        #                     print("3-{}:{}".format(k2,v2))

        #                     if(hasattr(v2,"items")):
        #                         for k3,v3 in v2.items():
        #                             print("4-{}:{}".format(k3,v3))

        self.conv1.set_weights(weights["Conv"])
        self.conv2.set_weights(weights["Conv_1"])
    
    def get_config(self):
        # config = super().get_config().copy()
        # config.update({
        #     'gap':self.gap,
        #     'conv1':self.conv1,
        #     'conv2':self.conv2,
        #     'act':self.act
        # })
        config = {
            'reduction':self.reduction,
            'l2_reg':self.l2_reg,
            # 'name':self.name,
        }
        # return config
        base_config = super(SEBottleneck, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LastStage(tflayers.Layer):
    def __init__(
            self,
            penultimate_channels: int,
            last_channels: int,
            num_classes: int,
            l2_reg: float,
    ):
        super().__init__(name="LastStage")
        self.penultimate_channels = penultimate_channels
        self.last_channels = last_channels
        self.num_classes = num_classes
        self.l2_reg = l2_reg

        self.conv1 = ConvNormAct(
            penultimate_channels,
            kernel_size=1,
            stride=1,
            norm_layer="bn",
            act_layer="hswish",
            use_bias=False,
            l2_reg=l2_reg,
            name="Conv_1",
        )
        self.gap = GlobalAveragePooling2D()
        self.conv2 = ConvNormAct(
            last_channels,
            kernel_size=1,
            norm_layer=None,
            act_layer="hswish",
            l2_reg=l2_reg,
            use_bias=False
        )
        self.dropout = tf.keras.layers.Dropout(
            rate=0.2,
            name="Dropout",
        )
        self.conv3 = tf.keras.layers.Dense(
            num_classes,
            # activation="softmax"
        )
        self.squeeze = Squeeze()

    def set_weights(self,weights):
        self.conv1.set_weights(weights[self.conv1.name])

    def call(self, input):
        x = self.conv1(input)
        # x = self.gap(x)
        # x = self.conv2(x)
        # x = self.dropout(x)
        # x = self.conv3(x)
        # x = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x))(x)
        # x = self.squeeze(x)
        return x

    def get_config(self):
        # config = super().get_config().copy()
        # config.update({
        #     'conv1':self.conv1
        # })
        config = {
            'penultimate_channels':self.penultimate_channels,
            'last_channels':self.last_channels,
            'num_classes':self.num_classes,
            'l2_reg':self.l2_reg,
        }
        # return config
        base_config = super(LastStage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))