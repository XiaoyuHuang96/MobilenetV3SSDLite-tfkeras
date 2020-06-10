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

MobileNetV3 Small
"""
import tensorflow as tf

from models.backbone.layers import ConvNormAct
from models.backbone.layers import Bneck
from models.backbone.layers import LastStage
from models.backbone.utils import _make_divisible
from models.backbone.utils import LayerNamespaceWrapper


class MobileNetV3(tf.keras.models.Model):
    def __init__(
            self,
            num_classes: int=1001,
            width_multiplier: float=1.0,
            name: str="MobileNetV3_Small",
            divisible_by: int=8,
            l2_reg: float=1e-5,
    ):
        super().__init__(name=name)

        # First layer
        self.first_layer = ConvNormAct(
            16,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_layer="bn",
            act_layer="hswish",
            use_bias=False,
            l2_reg=l2_reg,
            name="FirstLayer",
        )

        # Bottleneck layers
        self.bneck_settings = [
            # k   exp   out  SE      NL         s
            [ 3,  16,   16,  True,   "relu",    2 ],
            [ 3,  72,   24,  False,  "relu",    2 ],
            [ 3,  88,   24,  False,  "relu",    1 ],
            [ 5,  96,   40,  True,   "hswish",  2 ],
            [ 5,  240,  40,  True,   "hswish",  1 ],
            [ 5,  240,  40,  True,   "hswish",  1 ],
            [ 5,  120,  48,  True,   "hswish",  1 ],
            [ 5,  144,  48,  True,   "hswish",  1 ],
            [ 5,  288,  96,  True,   "hswish",  2 ],
            [ 5,  576,  96,  True,   "hswish",  1 ],
            [ 5,  576,  96,  True,   "hswish",  1 ],
        ]

        self.bneck = tf.keras.Sequential(name="Bneck")
        for idx, (k, exp, out, SE, NL, s) in enumerate(self.bneck_settings):
            out_channels = _make_divisible(out * width_multiplier, divisible_by)
            exp_channels = _make_divisible(exp * width_multiplier, divisible_by)

            self.bneck.add(
                LayerNamespaceWrapper(
                    Bneck(
                        out_channels=out_channels,
                        exp_channels=exp_channels,
                        kernel_size=k,
                        stride=s,
                        use_se=SE,
                        act_layer=NL,
                    ),
                    name=f"Bneck{idx}")
            )

        # Last stage
        penultimate_channels = _make_divisible(576 * width_multiplier, divisible_by)
        last_channels = _make_divisible(1_280 * width_multiplier, divisible_by)

        self.last_stage = LastStage(
            penultimate_channels,
            last_channels,
            num_classes,
            l2_reg=l2_reg,
        )

    def call(self, input):
        x = self.first_layer(input)
        x = self.bneck(x)
        x = self.last_stage(x)
        return x

def printlayerweights(layer):
    print(layer.name)
    for weight in layer.weights:
        print("weight name",weight.name, weight.shape)

class MobileNetV3Extractor(tf.keras.models.Model):
# class MobileNetV3Extractor(tf.keras.layers.Layer):
    def __init__(
            self,
            num_classes: int=1001,
            width_multiplier: float=1.0,
            name: str="MobilenetV3",#_Small",
            divisible_by: int=8,
            l2_reg: float=1e-5,
    ):
        super().__init__(name=name)

        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        # self.name = name
        self.divisible_by = divisible_by
        self.l2_reg = l2_reg

        # First layer
        self.first_layer = ConvNormAct(
            16,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_layer="bn",
            act_layer="hswish",
            use_bias=False,
            l2_reg=l2_reg,
            name="FirstLayer",
        )

        # Bottleneck layers
        self.bneck_settings = [
            # k   exp   out  SE      NL         s
            [ 3,  16,   16,  True,   "relu",    2 ],
            [ 3,  72,   24,  False,  "relu",    2 ],
            [ 3,  88,   24,  False,  "relu",    1 ],
            [ 5,  96,   40,  True,   "hswish",  2 ],
            [ 5,  240,  40,  True,   "hswish",  1 ],
            [ 5,  240,  40,  True,   "hswish",  1 ],
            [ 5,  120,  48,  True,   "hswish",  1 ],
            [ 5,  144,  48,  True,   "hswish",  1 ],
            # [ 5,  288,  96,  True,   "hswish",  2 ],
            [ 5,  288,  48,  True,   "hswish",  2 ],
            # [ 5,  576,  96,  True,   "hswish",  1 ],
            [ 5,  288,  48,  True,   "hswish",  1 ],
            # [ 5,  576,  96,  True,   "hswish",  1 ],
            [ 5,  288,  48,  True,   "hswish",  1 ],
        ]
        self.cls_ch_squeeze = 288 #576
        self.cls_ch_expand = 288 #1280

        # i=0
        self.bneck = [] #tf.keras.Sequential(name="Bneck")
        for idx, (k, exp, out, SE, NL, s) in enumerate(self.bneck_settings):
            out_channels = _make_divisible(out * width_multiplier, divisible_by)
            exp_channels = _make_divisible(exp * width_multiplier, divisible_by)
            has_expand = False
            if(idx == 8):
                has_expand = True
            convname = "expanded_conv"
            if(idx>0):
                convname += "_{}".format(idx)

            self.bneck.append(
                # LayerNamespaceWrapper(
                    Bneck(
                        out_channels=out_channels,
                        exp_channels=exp_channels,
                        kernel_size=k,
                        stride=s,
                        use_se=SE,
                        act_layer=NL,
                        has_expand=has_expand,
                        name=convname,
                    )#,
                    # name=f"Bneck{idx}")
            )
            # i+=1

        # Last stage
        penultimate_channels = _make_divisible( self.cls_ch_squeeze* width_multiplier, divisible_by)
        last_channels = _make_divisible(self.cls_ch_expand * width_multiplier, divisible_by)

        self.last_stage = LastStage(
            penultimate_channels,
            last_channels,
            num_classes,
            l2_reg=l2_reg,
        )

    def call(self, input):
        # print("input "+self.first_layer.name+"input",input.shape)
        x = self.first_layer(input)
        # printlayerweights(self.first_layer)
        # print("after "+self.first_layer.name,x.shape)

        # x = self.bneck(x)
        c4 = None
        for idx,conv in enumerate(self.bneck):
            if(idx==8):
                x,c4 = conv(x)
                # print("after "+conv.name+"-c4",c4.shape)
                # print("after "+conv.name+"-x",x.shape)
            else:
                x = conv(x)
            # printlayerweights(conv)
                # print("after "+conv.name,x.shape)
            # for weight in conv.weights:
                # print(weight.name, weight.shape)
        x = self.last_stage(x)
        # printlayerweights(self.last_stage)
        # print("after "+self.last_stage.name,x.shape)
        # for weight in self.last_stage.weights:
            # print(weight.name, weight.shape)
        return c4,x

    def get_config(self):

        # config = super().get_config().copy()
        # config.update({
        #     'first_layer':self.first_layer,
        #     'bneck_0':self.bneck[0],
        #     'bneck_1':self.bneck[1],
        #     'bneck_2':self.bneck[2],
        #     'bneck_3':self.bneck[3],
        #     'bneck_4':self.bneck[4],
        #     'bneck_5':self.bneck[5],
        #     'bneck_6':self.bneck[6],
        #     'bneck_7':self.bneck[7],
        #     'bneck_8':self.bneck[8],
        #     'bneck_9':self.bneck[9],
        #     'bneck_10':self.bneck[10],
        #     'last_stage':self.last_stage,
        # })
        config = {
            'num_classes':self.num_classes,
            'width_multiplier':self.width_multiplier,
            # 'name':self.name,
            'divisible_by':self.divisible_by,
            'l2_reg':self.l2_reg
        }
        # return config
        base_config = super(MobileNetV3Extractor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))