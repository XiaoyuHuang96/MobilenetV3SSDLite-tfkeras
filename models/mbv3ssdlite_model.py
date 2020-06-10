from __future__ import division
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.backend as K
from models.backbone.mobilenetv3_factory import build_mobilenetv3
# from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Activation, Conv2D, \
    DepthwiseConv2D, Reshape, Concatenate, BatchNormalization, ReLU
from layers.AnchorBoxesLayer import AnchorBoxes
from layers.DecodeDetectionsLayer import DecodeDetections
from layers.DecodeDetectionsFastLayer import DecodeDetectionsFast

def printlayerweights(layer):
    print("layer.name",layer.name)
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


def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

class ssdDownSampleBlock(tf.keras.layers.Layer):
# class ssdDownSampleBlock(tf.keras.models.Model):
    def __init__(self,conv_out_channel, sep_out_channel, id, model_type):
        super(ssdDownSampleBlock,self).__init__(name="ssdDownSampleBlock_{}".format(id))
        self.conv_out_channel = conv_out_channel
        self.sep_out_channel = sep_out_channel
        self.id = id
        self.model_type = model_type

        curlayerid = 13
        if(model_type=="large_extractor"):
            curlayerid=17
        self.name1 = 'layer_{}_1_Conv2d_{}_1x1_{}'.format(curlayerid,id,conv_out_channel)
        self.conv1 = KL.Conv2D(conv_out_channel, kernel_size=1, padding='same', use_bias=False,
                  activation=None, name=self.name1)
        self.bn1 = KL.BatchNormalization(epsilon=1e-3, momentum=0.999, name=self.name1+'/BatchNorm',trainable = False,fused=True)
        self.relu1 = KL.ReLU(6., name=self.name1 )

        self.name2 = 'layer_{}_2_Conv2d_{}_3x3_s2_{}_depthwise'.format(curlayerid,id,sep_out_channel)
        # name = 'SSD/2_Conv2d_{}_3x3_s2_{}_depthwise'.format(id,sep_out_channel)
        
        
        self.dwconv2d = KL.DepthwiseConv2D(kernel_size=3, strides=2,activation=None, use_bias=False, padding='same', name=self.name2)
                            # activation=None, use_bias=False, padding='valid', name=self.name2)
        self.bn2 = KL.BatchNormalization(epsilon=1e-3, momentum=0.999, name=self.name2+'/BatchNorm',trainable = False,fused=True)
        self.relu2 = KL.ReLU(6., name=self.name2)

        self.name3 = 'layer_{}_2_Conv2d_{}_3x3_s2_{}'.format(curlayerid,id,sep_out_channel)
        # name = 'SSD/3_Conv2d_{}_1x1_{}'.format(id,sep_out_channel)

        self.conv2 = KL.Conv2D(sep_out_channel, kernel_size=1, padding='same', use_bias=False,activation=None, name=self.name3)
        # x = KL.Conv2D(sep_out_channel, kernel_size=3, padding='same', use_bias=False,
                    
        self.bn3 = KL.BatchNormalization(epsilon=1e-3, momentum=0.999, name=self.name3+'/BatchNorm',trainable = False,fused=True)
        self.relu3 = KL.ReLU(6., name=self.name3)
    
    def call(self,inputs):
        # print("before "+self.conv1.name,inputs.shape)
        x = self.conv1(inputs)
        # printlayerweights(self.conv1)
        x = self.bn1(x)
        # printlayerweights(self.bn1)
        x = self.relu1(x)
        # print("after "+self.conv1.name,x.shape)
        
        # self.padding = KL.ZeroPadding2D(padding=correct_pad(K, x, 3), name=self.name2+'_zeropad')
        # x = self.padding(x)

        # print("after "+self.padding.name,x.shape)
        x = self.dwconv2d(x)
        # printlayerweights(self.dwconv2d)
        x = self.bn2(x)
        # printlayerweights(self.bn2)
        x = self.relu2(x)
        
        # print("after "+self.dwconv2d.name,x.shape)
        x = self.conv2(x)
        # printlayerweights(self.conv2)
        x = self.bn3(x)
        # printlayerweights(self.bn3)
        x = self.relu3(x)
        # print("after "+self.conv2.name,x.shape)
        return x
        
    def get_config(self):

        # config = super().get_config().copy()
        # config.update({
        #     'conv1':self.conv1,
        #     'bn1':self.bn1,
        #     'relu1':self.relu1,
        #     'padding':self.padding,
        #     'dwconv2d':self.dwconv2d,
        #     'bn2':self.bn2,
        #     'relu2':self.relu2,
        #     'conv2':self.conv2,
        #     'bn3':self.bn3,
        #     'relu3':self.relu3
        # })
        config = {
            'conv_out_channel':self.conv_out_channel,
            'sep_out_channel':self.sep_out_channel,
            'id':self.id,
            'model_type':self.model_type,
        }
        # return config
        base_config = super(ssdDownSampleBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def set_weights(self,weights):
        # conv1
        curweights = weights[self.conv1.name]

        conv1_dicts = {"kernel:0"}
        conv1_weights = []
        for key in conv1_dicts:
            conv1_weights.append(curweights[key])
        self.conv1.set_weights(conv1_weights)

        # bn1
        bn_dicts = ["gamma:0","beta:0","moving_mean:0","moving_variance:0"]
        bn1_weights = []
        
        for key in bn_dicts:
            # print("bn keys",key)
            bn1_weights.append(curweights["BatchNorm"][key])
        self.bn1.set_weights(bn1_weights)

        # dep conv2
        curweights = weights[self.dwconv2d.name]
        dwconv2_dicts = {"depthwise_kernel:0"}
        dwconv2_weights = []
        for key in dwconv2_dicts:
            dwconv2_weights.append(curweights[key])
        self.dwconv2d.set_weights(dwconv2_weights)

        # bn2
        bn2_weights = []
        
        for key in bn_dicts:
            # print("bn keys",key)
            bn2_weights.append(curweights["BatchNorm"][key])
        self.bn2.set_weights(bn2_weights)

        # conv2
        curweights = weights[self.conv2.name]

        conv2_dicts = {"kernel:0"}
        conv2_weights = []
        for key in conv2_dicts:
            conv2_weights.append(curweights[key])
        self.conv2.set_weights(conv2_weights)

        # bn3
        bn3_weights = []
        
        for key in bn_dicts:
            # print("bn keys",key)
            bn3_weights.append(curweights["BatchNorm"][key])
        self.bn3.set_weights(bn3_weights)       


        

class MobileNetV3SSDLiteExtractor(tf.keras.models.Model):
    def __init__(
            self,
            model_type:str="small",
            height:int=300,
            width:int=300,
            channels:int=3,
            num_classes: int=1001,
            width_multiplier: float=1.0,
            name: str='FeatureExtractor',
            divisible_by: int=8,
            l2_reg: float=1e-5,
    ):
        # print(name)
        super(MobileNetV3SSDLiteExtractor,self).__init__(name=name)
        
        self.model_type = model_type
        self.height = height
        self.width = width
        self.channels = channels
        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        # self.name = name
        self.divisible_by = divisible_by
        self.l2_reg = l2_reg

        self.backbonenet = build_mobilenetv3(
            model_type,
            input_shape=(height, width, channels),
            num_classes=num_classes,
            width_multiplier=width_multiplier,
            l2_reg=l2_reg,
        )

        # self.relu1 = KL.ReLU(6.,name="ssd_2_conv_relu")
        self.conv3 = ssdDownSampleBlock(256,512,2,model_type)
        self.conv4 = ssdDownSampleBlock(128,256,3,model_type)
        self.conv5 = ssdDownSampleBlock(128,256,4,model_type)
        self.conv6 = ssdDownSampleBlock(64,128,5,model_type)
        
    # def build(self, input_shape):
    #     self.in_channels = int(input_shape[3])
    #     super().build(input_shape)

    def call(self,inputs):
        # print("backbone input ",inputs.shape)
        x1, x = self.backbonenet(inputs)
        
        # printlayerweights(self.backbonenet)
        # print("backbone",self.backbonenet.summary())
        # print("after "+self.backbonenet.name,x1.shape)
        x2 = x #self.relu1(x)
        # print("after "+self.relu1.name,x2.shape)
        x3 = self.conv3(x2)
        # printlayerweights(self.conv3)
        # print("after "+self.conv3.name,x3.shape)
        x4 = self.conv4(x3)
        # printlayerweights(self.conv4)
        # print("after "+self.conv4.name,x4.shape)
        x5 = self.conv5(x4)
        # printlayerweights(self.conv5)
        # print("after "+self.conv5.name,x5.shape)
        x6 = self.conv6(x5)
        # printlayerweights(self.conv6)
        # print("after "+self.conv6.name,x6.shape)
        return [x1,x2,x3,x4,x5,x6]
    
    def get_config(self):

        # config = super().get_config().copy()
        # config.update({
        #     'backbonenet':self.backbonenet,
        #     'relu1':self.relu1,
        #     'conv3':self.conv3,
        #     'conv4':self.conv4,
        #     'conv5':self.conv5,
        #     'conv6':self.conv6,
        # })
        config = {
            'model_type':self.model_type,
            'height':self.height,
            'width':self.width,
            'channels':self.channels,
            'num_classes':self.num_classes,
            'width_multiplier':self.width_multiplier,
            # 'name':self.name,
            'divisible_by':self.divisible_by,
            'l2_reg':self.l2_reg,
        }
        # return config
        base_config = super(MobileNetV3SSDLiteExtractor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class PredictBlock(tf.keras.layers.Layer):
# class PredictBlock(tf.keras.models.Model):
    def __init__(self,out_channel, sym, id):
        self.out_channel = out_channel
        self.sym = sym
        self.id = id 
        name = 'BoxPredictor_' + '{}_'.format(id) + sym
        super().__init__(name=name )
        name = sym
        self.dwconv2d = DepthwiseConv2D(kernel_size=3, strides=1,
                           activation=None, use_bias=False, padding='same', name=  name + '{}_depthwise'.format(id))
        self.bn = BatchNormalization(epsilon=1e-3, momentum=0.999, name= name + '{}_depthwise/BatchNorm'.format(id),trainable = False,fused=True)
        self.relu = ReLU(6., name=name + '{}_depthwise/ReLU'.format(id))

        # self.conv2d = Conv2D(out_channel, kernel_size=1, padding='same', use_bias=False,
        #           activation=None, name=name+'{}_conv1'.format(id))
        self.conv2d = Conv2D(out_channel, kernel_size=1, padding='same', use_bias=True,
                  activation=None, name=name+'{}_conv1'.format(id))
        # x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '/conv2_bn')(x)
        # self.bn2 = BatchNormalization(epsilon=1e-3, momentum=0.999, name= name + '{}_depthwise/BatchNorm2'.format(id))
    
    def call(self,inputs,training=False):
        # print("before "+self.dwconv2d.name,inputs.shape)
        x = self.dwconv2d(inputs)
        # printlayerweights(self.dwconv2d)
        # print("after "+self.dwconv2d.name,x.shape)
        x = self.bn(x)
        # printlayerweights(self.bn)
        x = self.relu(x)
        x = self.conv2d(x)
        # x = self.bn2(x)
        # printlayerweights(self.conv2d)
        # print("after "+self.conv2d.name,x.shape)
        return x

    # def build(self, input_shape):
    #     self.in_channels = int(input_shape[3])
    #     super().build(input_shape)
    
    def set_weights(self,weights):
        
        # for key,value in weights.items():
        #     print(key,value)
        #     print("loadding trees")
        #     for k,v in value.items():
        #         print(k,v)
        #     print("loadded")

        # depth wiseconv
        curweights = weights[self.dwconv2d.name]
        dwconv2_dicts = {"depthwise_kernel:0"}
        dwconv2_weights = []
        for key in dwconv2_dicts:
            dwconv2_weights.append(curweights[key])
        self.dwconv2d.set_weights(dwconv2_weights)

        # bn2
        bn_dicts = ["gamma:0","beta:0","moving_mean:0","moving_variance:0"]
        bn_weights = []
        
        for key in bn_dicts:
            # print("bn keys",key)
            bn_weights.append(curweights["BatchNorm"][key])
        self.bn.set_weights(bn_weights)

        # conv2
        curweights = weights[self.conv2d.name]

        conv2d_dicts = [] #{"kernel:0","biases:0"}
        origin_w = self.conv2d.get_weights()
        for i in range(len(origin_w)):
            # print(i,len(np.array(origin_w[i]).shape))
            if(len(np.array(origin_w[i]).shape)>2):
                conv2d_dicts.append("kernel:0")
            else:
                conv2d_dicts.append("biases:0")
        conv2d_weights = []
        for key in conv2d_dicts:
            conv2d_weights.append(curweights[key])
        
        
        self.conv2d.set_weights(conv2d_weights)


        # bn_dicts = ["gamma:0","beta:0","moving_mean:0","moving_variance:0"]
        # norm_weights = []
        
        # for key in bn_dicts:
        #     # print("bn keys",key)
        #     norm_weights.append(weights["BatchNorm"][key])
        # self.norm.set_weights(norm_weights)

    
    def get_config(self):
        # config = super().get_config().copy()
        # config.update({
        #     'dwconv2d':self.dwconv2d,
        #     'bn':self.bn,
        #     'relu':self.relu,
        #     'conv2d':self.conv2d
        # })
        config = {
            'out_channel':self.out_channel,
            'sym':self.sym,
            'id':self.id, 
        }
        base_config = super(PredictBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        # return config

class MobileNetV3SSDLite(tf.keras.models.Model):
    def __init__(self,
            image_size,
            n_classes,
            mode='training',
            l2_regularization=0.0005,
            min_scale=None,
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],
            two_boxes_for_ar1=True,
            steps=[8, 16, 32, 64, 100, 300],
            offsets=None,
            clip_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            coords='centroids',
            normalize_coords=True,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=[2, 1, 0],
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            nms_max_output_size=400,
            return_predictor_sizes=False,
            model_type="small_extractor",
            width_multiplier=1.0,
            model_name="Mobilenetv3_small",
            divisible_by=8):
        """
        Build a Keras model with SSD300 architecture, see references.
        The base network is a reduced atrous VGG-16, extended by the SSD architecture,
        as described in the paper.
        Most of the arguments that this function takes are only needed for the anchor
        box layers. In case you're training the network, the parameters passed here must
        be the same as the ones used to set up `SSDBoxEncoder`. In case you're loading
        trained weights, the parameters passed here must be the same as the ones used
        to produce the trained weights.
        Some of these arguments are explained in more detail in the documentation of the
        `SSDBoxEncoder` class.
        Note: Requires Keras v2.0 or later. Currently works only with the
        TensorFlow backend (v1.0 or later).
        Arguments:
            image_size (tuple): The input image size in the format `(height, width, channels)`.
            n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
            mode (str, optional): One of 'training', 'inference' and 'inference_fast'. In 'training' mode,
                the model outputs the raw prediction tensor, while in 'inference' and 'inference_fast' modes,
                the raw predictions are decoded into absolute coordinates and filtered via confidence thresholding,
                non-maximum suppression, and top-k filtering. The difference between latter two modes is that
                'inference' follows the exact procedure of the original Caffe implementation, while
                'inference_fast' uses a faster prediction decoding procedure.
            l2_regularization (float, optional): The L2-regularization rate. Applies to all convolutional layers.
                Set to zero to deactivate L2-regularization.
            min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
                of the shorter side of the input images.
            max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
                of the shorter side of the input images. All scaling factors between the smallest and the
                largest will be linearly interpolated. Note that the second to last of the linearly interpolated
                scaling factors will actually be the scaling factor for the last predictor layer, while the last
                scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
                if `two_boxes_for_ar1` is `True`.
            scales (list, optional): A list of floats containing scaling factors per convolutional predictor layer.
                This list must be one element longer than the number of predictor layers. The first `k` elements are the
                scaling factors for the `k` predictor layers, while the last element is used for the second box
                for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
                last scaling factor must be passed either way, even if it is not being used. If a list is passed,
                this argument overrides `min_scale` and `max_scale`. All scaling factors must be greater than zero.
            aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are to be
                generated. This list is valid for all prediction layers.
            aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each prediction layer.
                This allows you to set the aspect ratios for each predictor layer individually, which is the case for the
                original SSD300 implementation. If a list is passed, it overrides `aspect_ratios_global`.
            two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratio lists that contain 1. Will be ignored
                otherwise. If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
                using the scaling factor for the respective layer, the second one will be generated using
                geometric mean of said scaling factor and next bigger scaling factor.
            steps (list, optional): `None` or a list with as many elements as there are predictor layers. The elements
                can be either ints/floats or tuples of two ints/floats. These numbers represent for each predictor layer
                how many pixels apart the anchor box center points should be vertically and horizontally along the spatial
                grid over the image. If the list contains ints/floats, then that value will be used for both spatial
                dimensions. If the list contains tuples of two ints/floats, then they represent `(step_height, step_width)`.
                If no steps are provided, then they will be computed such that the anchor box center points will form an
                equidistant grid within the image dimensions.
            offsets (list, optional): `None` or a list with as many elements as there are predictor layers. The elements
                can be either floats or tuples of two floats. These numbers represent for each predictor layer how many
                pixels from the top and left boarders of the image the top-most and left-most anchor box center points
                should be as a fraction of `steps`. The last bit is important: The offsets are not absolute pixel values,
                but fractions of the step size specified in the `steps` argument. If the list contains floats, then that
                value will be used for both spatial dimensions. If the list contains tuples of two floats, then they
                represent `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will default
                to 0.5 of the step size.
            clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within image boundaries.
            variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
                its respective variance value.
            coords (str, optional): The box coordinate format to be used internally by the model (i.e. this is not the
                input format of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)`
                (box center coordinates, width, and height), 'minmax' for the format `(xmin, xmax, ymin, ymax)`,
                or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
            normalize_coords (bool, optional): Set to `True` if the model is supposed to use relative instead of
                absolute coordinates, i.e. if the model predicts box coordinates within [0,1] instead of absolute
                coordinates.
            subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
                of any shape that is broadcast-compatible with the image shape. The elements of this array will be
                subtracted from the image pixel intensity values. For example, pass a list of three integers
                to perform per-channel mean normalization for color images.
            divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
                floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
                intensity values will be divided by the elements of this array. For example, pass a list
                of three integers to perform per-channel standard deviation normalization for color images.
            swap_channels (list, optional): Either `False` or a list of integers representing the desired order in which
                the input image channels should be swapped.
            confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
                positive class in order to be considered for the non-maximum suppression stage for the respective class.
                A lower value will result in a larger part of the selection process being done by the non-maximum
                suppression stage, while a larger value will result in a larger part of the selection process happening
                in the confidence thresholding stage.
            iou_threshold (float, optional): A float in [0,1]. All boxes that have a Jaccard similarity of greater
                than `iou_threshold` with a locally maximal box will be removed from the set of predictions for a given
                class, where 'maximal' refers to the box's confidence score.
            top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
                non-maximum suppression stage.
            nms_max_output_size (int, optional): The maximal number of predictions that will be left over after the
                NMS stage.
            return_predictor_sizes (bool, optional): If `True`, this function not only returns the model, but also
                a list containing the spatial dimensions of the predictor layers. This isn't strictly necessary since
                you can always get their sizes easily via the Keras API, but it's convenient and less error-prone
                to get them this way. They are only relevant for training anyway (SSDBoxEncoder needs to know the
                spatial dimensions of the predictor layers), for inference you don't need them.
        Returns:
            model: The Keras SSD300 model.
            predictor_sizes (optional): A Numpy array containing the `(height, width)` portion
                of the output tensor shape for each convolutional predictor layer. During
                training, the generator function needs this in order to transform
                the ground truth labels into tensors of identical structure as the
                output tensors of the model, which is in turn needed for the cost
                function.
        References:
            https://arxiv.org/abs/1512.02325v5
        """
        super().__init__(name="SSDLite")

        self.image_size = image_size
        self.n_classes = n_classes
        self.mode = mode
        self.l2_regularization = l2_regularization
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scales = scales
        self.aspect_ratios_global = aspect_ratios_global
        self.aspect_ratios_per_layer = aspect_ratios_per_layer
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.steps = steps
        self.offsets = offsets
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.coords = coords
        self.normalize_coords = normalize_coords
        self.subtract_mean = subtract_mean
        self.divide_by_stddev = divide_by_stddev
        self.swap_channels = swap_channels
        self.confidence_thresh = confidence_thresh
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.nms_max_output_size = nms_max_output_size
        self.return_predictor_sizes = return_predictor_sizes
        self.model_type = model_type
        self.width_multiplier = width_multiplier
        self.model_name = model_name
        self.divisible_by = divisible_by

        n_predictor_layers = 6  # The number of predictor conv layers in the network is 6 for the original SSD300.
        n_classes += 1  # Account for the background class.
        l2_reg = l2_regularization  # Make the internal name shorter.
        img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]
        
        # args={"model_type":model_type,"height":img_height,"width":img_width,"channels":img_channels,"num_classes":n_classes,
        #     "width_multiplier":width_multiplier,"l2_reg":l2_regularization}

        ############################################################################
        # Get a few exceptions out of the way.
        ############################################################################

        if aspect_ratios_global is None and aspect_ratios_per_layer is None:
            raise ValueError(
                "`aspect_ratios_global` and `aspect_ratios_per_layer` \
                cannot both be None. At least one needs to be specified.")
        if aspect_ratios_per_layer:
            if len(aspect_ratios_per_layer) != n_predictor_layers:
                raise ValueError(
                    "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, \
                    but len(aspect_ratios_per_layer) == {}.".format(
                        n_predictor_layers, len(aspect_ratios_per_layer)))

        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
        if scales:
            if len(scales) != n_predictor_layers + 1:
                raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                    n_predictor_layers + 1, len(scales)))
        # If no explicit list of scaling factors was passed,
        # compute the list of scaling factors from `min_scale` and `max_scale`
        else:
            scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        if (not (steps is None)) and (len(steps) != n_predictor_layers):
            raise ValueError("You must provide at least one step value per predictor layer.")

        if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
            raise ValueError("You must provide at least one offset value per predictor layer.")

        ############################################################################
        # Compute the anchor box parameters.
        ############################################################################

        # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
        if aspect_ratios_per_layer:
            aspect_ratios = aspect_ratios_per_layer
        else:
            aspect_ratios = [aspect_ratios_global] * n_predictor_layers

        # Compute the number of boxes to be predicted per cell for each predictor layer.
        # We need this so that we know how many channels the predictor layers need to have.
        if aspect_ratios_per_layer:
            n_boxes = []
            for ar in aspect_ratios_per_layer:
                if (1 in ar) & two_boxes_for_ar1:
                    n_boxes.append(len(ar) + 1)  # +1 for the second box for aspect ratio 1
                else:
                    n_boxes.append(len(ar))
        # If only a global aspect ratio list was passed,
        # then the number of boxes is the same for each predictor layer
        else:
            if (1 in aspect_ratios_global) & two_boxes_for_ar1:
                n_boxes = len(aspect_ratios_global) + 1
            else:
                n_boxes = len(aspect_ratios_global)
            n_boxes = [n_boxes] * n_predictor_layers

        if steps is None:
            steps = [None] * n_predictor_layers
        if offsets is None:
            offsets = [None] * n_predictor_layers

        ############################################################################
        # Define functions for the Lambda layers below.
        ############################################################################

        def identity_layer(tensor):
            return tensor

        def input_mean_normalization(tensor):
            return tensor - np.array(subtract_mean)

        def input_stddev_normalization(tensor):
            return tensor / np.array(divide_by_stddev)

        def input_channel_swap(tensor):
            if len(swap_channels) == 3:
                return K.stack(
                    [tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]]], axis=-1)
            elif len(swap_channels) == 4:
                return K.stack([tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]],
                                tensor[..., swap_channels[3]]], axis=-1)

        ############################################################################
        # Build the network.
        ############################################################################

        # x = Input(shape=(img_height, img_width, img_channels))

        # The following identity layer is only needed so that the subsequent lambda layers can be optional.
        self.lambda_layer = Lambda(identity_layer, input_shape=(img_height, img_width, img_channels),output_shape=(img_height, img_width, img_channels), name='identity_layer')#(x)

        # self.tmp_shape = K.int_shape(x1)
        self.subtract_mean = subtract_mean
        if not (subtract_mean is None):
            self.subtract_mean_layer = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels),
                        name='input_mean_normalization')
        self.divide_by_stddev = divide_by_stddev
        if not (divide_by_stddev is None):
            self.divide_by_stddev_layer = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels),
                        name='input_stddev_normalization')
        
        self.swap_channels = swap_channels
        self.swap_channels_layer = None
        if swap_channels:
            self.swap_channels_layer = Lambda(
                input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')

        self.extractor = MobileNetV3SSDLiteExtractor(model_type,img_height,img_width,img_channels,n_classes,width_multiplier,
            model_name,divisible_by,l2_regularization)
        
        self.n_predictor_layers = n_predictor_layers
        self.cls_layers = []
        self.box_layers = []
        self.priorboxs = []
        self.cls_reshapes = []
        self.box_reshapes = []
        self.priorboxs_reshapes = []
        for i in range(n_predictor_layers):
            # print("predict cls num:",i,n_boxes[i]*n_classes)
            self.cls_layers.append(PredictBlock(n_boxes[i]*n_classes,'ClassPredictor',i))
            self.box_layers.append(PredictBlock(n_boxes[i]*4,'BoxEncodingPredictor',i))
            self.priorboxs.append(AnchorBoxes(img_height, img_width, this_scale=scales[i], next_scale=scales[i+1],
                                             aspect_ratios=aspect_ratios[i],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[i],
                                             this_offsets=offsets[i], clip_boxes=clip_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords,
                                             name='BoxPredictor_{}'.format(i)))
            self.cls_reshapes.append(Reshape((-1, n_classes), name='ssd_cls{}_reshape'.format(i)))
            self.box_reshapes.append(Reshape((-1, 4), name='ssd_box{}_reshape'.format(i)))
            self.priorboxs_reshapes.append(Reshape((-1, 4), name='ssd_priorbox{}_reshape'.format(i)))

        self.concat_cls = Concatenate(axis=1, name='ssd_cls')
        self.concat_box = Concatenate(axis=1, name='ssd_box')
        self.concat_priorbox = Concatenate(axis=1, name='ssd_priorbox')
        self.act_cls = Activation('sigmoid', name='ssd_mbox_conf_sigmoid')
        # self.act_cls = Activation('softmax', name='ssd_mbox_conf_softmax')
        self.concat_preds = Concatenate(axis=2, name='ssd_predictions')

        self.mode = mode

        if self.mode == 'inference':
            self.decoder = DecodeDetections(confidence_thresh=confidence_thresh,
                                                iou_threshold=iou_threshold,
                                                top_k=top_k,
                                                nms_max_output_size=nms_max_output_size,
                                                coords=coords,
                                                normalize_coords=normalize_coords,
                                                img_height=img_height,
                                                img_width=img_width,
                                                name='ssd_decoded_predictions')
        elif self.mode == 'inference_fast':
            self.decoder = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                    iou_threshold=iou_threshold,
                                                    top_k=top_k,
                                                    nms_max_output_size=nms_max_output_size,
                                                    coords=coords,
                                                    normalize_coords=normalize_coords,
                                                    img_height=img_height,
                                                    img_width=img_width,
                                                    name='ssd_decoded_predictions')

    # def build(self, input_shape):
    #     self.in_channels = int(input_shape[3])
    #     super().build(input_shape)

    def call(self,inputs,training=False):
        x = self.lambda_layer(inputs)
        if not (self.subtract_mean is None):
            x = self.subtract_mean_layer(x)
        if not (self.divide_by_stddev is None):
            x = self.divide_by_stddev_layer(x)
        if self.swap_channels:
            x = self.swap_channels_layer(x)

        # x = x*(2.0 / 255.0)-1.0
        
        # print("before "+self.extractor.name,x.shape)
        links = self.extractor(x)
        # print("self.extractor",self.extractor.summary())
        print("links",links)
        
        cls_feas = []
        box_feas = []
        priorboxs_feas = []
        for i in range(self.n_predictor_layers):
            # print("before "+self.cls_layers[i].name,links[i].shape)
            cls_fea = self.cls_layers[i](links[i])
            # print("after "+self.cls_layers[i].name,cls_fea.shape)
            cls_fea = self.cls_reshapes[i](cls_fea)
            # print("after "+self.cls_reshapes[i].name,cls_fea.shape)
            cls_feas.append(cls_fea)

            box_fea = self.box_layers[i](links[i])
            # print("after "+self.box_layers[i].name,box_fea.shape)
            box_fea_reshape = self.box_reshapes[i](box_fea)
            # print("after "+self.box_reshapes[i].name,box_fea_reshape.shape)
            box_feas.append(box_fea_reshape)

            print("box num",i,box_fea.shape)

            priorbox_fea = self.priorboxs[i](box_fea)
            # print("after "+self.priorboxs[i].name,priorbox_fea.shape)
            priorbox_fea = self.priorboxs_reshapes[i](priorbox_fea)
            # print("after "+self.priorboxs_reshapes[i].name,priorbox_fea.shape)
            priorboxs_feas.append(priorbox_fea)


        clss = self.concat_cls(cls_feas)
        # print("after "+self.concat_cls.name,clss.shape)
        box = self.concat_box(box_feas)
        # print("after "+self.concat_box.name,box.shape)
        priorbox = self.concat_priorbox(priorboxs_feas)
        # print("after "+self.concat_priorbox.name,priorbox.shape)

        clss = self.act_cls(clss)
        preds =  self.concat_preds([clss, box, priorbox])
        # print("after "+self.concat_preds.name,preds.shape)

        print("preds",preds)

        if self.mode == 'training':
            return preds
        elif self.mode == 'inference' or self.mode == 'inference_fast':
            # return preds
            outputs = self.decoder(preds)
            # print("after "+self.decoder.name,outputs.shape)
            return outputs
        else:
            raise ValueError(
                "`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(self.mode))

        # if return_predictor_sizes:
        #     predictor_sizes = np.array(
        #         [cls[0]._keras_shape[1:3], cls[1]._keras_shape[1:3], cls[2]._keras_shape[1:3],
        #         cls[3]._keras_shape[1:3], cls[4]._keras_shape[1:3], cls[5]._keras_shape[1:3]])
        #     return model, predictor_sizes
        # else:
        #     return model

    def get_config(self):
        # config = super().get_config().copy()
        # config.update({
        #     'image_size': self.image_size,
        #     'lambda_layer': self.lambda_layer,
        #     'subtract_mean_layer': self.subtract_mean_layer,
        #     'divide_by_stddev_layer': self.divide_by_stddev_layer,
        #     'swap_channels_layer': self.swap_channels_layer,
        #     'extractor': self.extractor,
        #     'cls_layers_0':self.cls_layers[0],
        #     'cls_layers_1':self.cls_layers[1],
        #     'cls_layers_2':self.cls_layers[2],
        #     'cls_layers_3':self.cls_layers[3],
        #     'cls_layers_4':self.cls_layers[4],
        #     'cls_layers_5':self.cls_layers[5],
        #     'box_layers_0':self.box_layers[0],
        #     'box_layers_1':self.box_layers[1],
        #     'box_layers_2':self.box_layers[2],
        #     'box_layers_3':self.box_layers[3],
        #     'box_layers_4':self.box_layers[4],
        #     'box_layers_5':self.box_layers[5],
        #     'priorboxs_0':self.priorboxs[0],
        #     'priorboxs_1':self.priorboxs[1],
        #     'priorboxs_2':self.priorboxs[2],
        #     'priorboxs_3':self.priorboxs[3],
        #     'priorboxs_4':self.priorboxs[4],
        #     'priorboxs_5':self.priorboxs[5],
        #     'cls_reshapes_0':self.cls_reshapes[0],
        #     'cls_reshapes_1':self.cls_reshapes[1],
        #     'cls_reshapes_2':self.cls_reshapes[2],
        #     'cls_reshapes_3':self.cls_reshapes[3],
        #     'cls_reshapes_4':self.cls_reshapes[4],
        #     'cls_reshapes_5':self.cls_reshapes[5],
        #     'box_reshapes_0':self.box_reshapes[0],
        #     'box_reshapes_1':self.box_reshapes[1],
        #     'box_reshapes_2':self.box_reshapes[2],
        #     'box_reshapes_3':self.box_reshapes[3],
        #     'box_reshapes_4':self.box_reshapes[4],
        #     'box_reshapes_5':self.box_reshapes[5],
        #     'priorboxs_reshapes_0':self.priorboxs_reshapes[0],
        #     'priorboxs_reshapes_1':self.priorboxs_reshapes[1],
        #     'priorboxs_reshapes_2':self.priorboxs_reshapes[2],
        #     'priorboxs_reshapes_3':self.priorboxs_reshapes[3],
        #     'priorboxs_reshapes_4':self.priorboxs_reshapes[4],
        #     'priorboxs_reshapes_5':self.priorboxs_reshapes[5],
        #     'concat_cls':self.concat_cls,
        #     'concat_box':self.concat_box,
        #     'concat_priorbox':self.concat_priorbox,
        #     'act_cls':self.act_cls,
        #     'concat_preds':self.concat_preds,
        #     'decoder':self.decoder,
        # })
        config = {
            'image_size':self.image_size,
            'n_classes':self.n_classes,
            'mode':self.mode,
            'l2_regularization':self.l2_regularization,
            'min_scale':self.min_scale,
            'max_scale':self.max_scale,
            'scales':self.scales,
            'aspect_ratios_global':self.aspect_ratios_global,
            'aspect_ratios_per_layer':self.aspect_ratios_per_layer,
            'two_boxes_for_ar1':self.two_boxes_for_ar1,
            'steps':self.steps,
            'offsets':self.offsets,
            'clip_boxes':self.clip_boxes,
            'variances':self.variances,
            'coords':self.coords,
            'normalize_coords':self.normalize_coords,
            'subtract_mean':self.subtract_mean,
            'divide_by_stddev':self.divide_by_stddev,
            'swap_channels':self.swap_channels,
            'confidence_thresh':self.confidence_thresh,
            'iou_threshold':self.iou_threshold,
            'top_k':self.top_k,
            'nms_max_output_size':self.nms_max_output_size,
            'return_predictor_sizes':self.return_predictor_sizes,
            'model_type':self.model_type,
            'width_multiplier':self.width_multiplier,
            'model_name':self.model_name,
            'divisible_by':self.divisible_by,
        }
        base_config = super(MobileNetV3SSDLite, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        # return config



