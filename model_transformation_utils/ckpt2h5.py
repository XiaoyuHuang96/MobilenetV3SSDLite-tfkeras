import tensorflow as tf
import h5py
from tensorflow.keras.utils import get_file, Progbar
from tensorflow.keras import Input, Model, layers
import pprint
import tensorflow.keras.backend as K
from models.backbone.mobilenetv3_large import MobileNetV3Extractor

# import deepdish as dd
import argparse
import os
import numpy as np

class MyCallbacks(tf.keras.callbacks.Callback):
    def __init__(self, pretrained_file):
        self.pretrained_file = pretrained_file
        self.sess = tf.keras.backend.get_session()
        self.saver = tf.train.Saver()
    def on_train_begin(self, logs=None):
        if self.pretrian_model_path:
            self.saver.restore(self.sess, self.pretrian_model_path)
            print('load weights: OK.')

def load_weights_from_tf_checkpoint(model, checkpoint_file, background_label):
    print('Load weights from tensorflow checkpoint')
    progbar = Progbar(target=len(model.layers))
 
    reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_file)
    
    # pprint.pprint(reader.debug_string().decode("utf-8")) #类型是str
    
    for index, layer in enumerate(model.layers):
        progbar.update(current=index)
        
        print("layer.name",layer.name)
        print("layer",layer)

        if isinstance(layers,MobileNetV3Extractor):
            for index, llayers in enumerate(layers.layers):
                progbar.update(current=index)
        
                print("llayers.name",llayers.name)

                if isinstance(llayers, layers.SeparableConv2D):
                    depthwise = reader.get_tensor('{}/depthwise_weights'.format(llayers.name))
                    pointwise = reader.get_tensor('{}/pointwise_weights'.format(llayers.name))
        
                    if K.image_data_format() == 'channels_first':
                        depthwise = convert_kernel(depthwise)
                        pointwise = convert_kernel(pointwise)
        
                    llayers.set_weights([depthwise, pointwise])
                elif isinstance(llayers, layers.Conv2D):
                    weights = reader.get_tensor('{}/weights'.format(llayers.name))
        
                    if K.image_data_format() == 'channels_first':
                        weights = convert_kernel(weights)
        
                    llayers.set_weights([weights])
                elif isinstance(llayers, layers.BatchNormalization):
                    beta = reader.get_tensor('{}/beta'.format(llayers.name))
                    gamma = reader.get_tensor('{}/gamma'.format(llayers.name))
                    moving_mean = reader.get_tensor('{}/moving_mean'.format(llayers.name))
                    moving_variance = reader.get_tensor('{}/moving_variance'.format(llayers.name))
        
                    llayers.set_weights([gamma, beta, moving_mean, moving_variance])
                elif isinstance(llayers, layers.Dense):
                    weights = reader.get_tensor('{}/weights'.format(llayers.name))
                    biases = reader.get_tensor('{}/biases'.format(llayers.name))
        
                    if background_label:
                        llayers.set_weights([weights, biases])
                    else:
                        llayers.set_weights([weights[:, 1:], biases[1:]])

        elif isinstance(layer, layers.SeparableConv2D):
            depthwise = reader.get_tensor('{}/depthwise_weights'.format(layer.name))
            pointwise = reader.get_tensor('{}/pointwise_weights'.format(layer.name))
 
            if K.image_data_format() == 'channels_first':
                depthwise = convert_kernel(depthwise)
                pointwise = convert_kernel(pointwise)
 
            layer.set_weights([depthwise, pointwise])
        elif isinstance(layer, layers.Conv2D):
            weights = reader.get_tensor('{}/weights'.format(layer.name))
 
            if K.image_data_format() == 'channels_first':
                weights = convert_kernel(weights)
 
            layer.set_weights([weights])
        elif isinstance(layer, layers.BatchNormalization):
            beta = reader.get_tensor('{}/beta'.format(layer.name))
            gamma = reader.get_tensor('{}/gamma'.format(layer.name))
            moving_mean = reader.get_tensor('{}/moving_mean'.format(layer.name))
            moving_variance = reader.get_tensor('{}/moving_variance'.format(layer.name))
 
            layer.set_weights([gamma, beta, moving_mean, moving_variance])
        elif isinstance(layer, layers.Dense):
            weights = reader.get_tensor('{}/weights'.format(layer.name))
            biases = reader.get_tensor('{}/biases'.format(layer.name))
 
            if background_label:
                layer.set_weights([weights, biases])
            else:
                layer.set_weights([weights[:, 1:], biases[1:]])
 
 
def load_pretrained_weights(model, fname, origin, md5_hash, background_label=False, cache_dir=None):
    """Download and convert tensorflow checkpoints"""
    # origin = 'https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz'
 
    if cache_dir is None:
        cache_dir = os.path.expanduser(os.path.join('~', '.keras', 'models'))
 
    weight_path = os.path.join(cache_dir, '{}_{}_{}.h5'.format(model.name, md5_hash, K.image_data_format()))
 
    if os.path.exists(weight_path):
        model.load_weights(weight_path)
    else:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            print(cache_dir)
        path = get_file(fname, origin=origin, extract=True, md5_hash=md5_hash, cache_dir=cache_dir,cache_subdir='.')
        checkpoint_file = os.path.join(path, '..', 'model.ckpt')
        load_weights_from_tf_checkpoint(model, checkpoint_file, background_label)
        model.save_weights(weight_path)


# load_pretrained_weights(model, fname, origin, md5_hash, background_label=False, cache_dir=None)


def tr(v):
    # tensorflow weights to pytorch weights
    if v.ndim == 4:
        return np.ascontiguousarray(v.transpose(3,2,0,1))
    elif v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v

def read_ckpt(ckpt):
    # https://github.com/tensorflow/tensorflow/issues/1823
    reader = tf.train.NewCheckpointReader(ckpt)
    weights = {n: reader.get_tensor(n) for (n, _) in reader.get_variable_to_shape_map().items()}
    pyweights = {k: tr(v) for (k, v) in weights.items()}
    return pyweights


ckpt2kerasdict={
    "FeatureExtractor/MobilenetV3/Conv": "MobileNetV3_large/FirstLayer",
    "FeatureExtractor/MobilenetV3/Conv_1": "MobileNetV3_large/LastStage",
}


def isKeyValid(key):
    validset = {"biases","weights","depthwise_weights","beta","gamma","moving_mean","moving_variance"}
    keysplits = key.split(r'/')
    if keysplits[-1] in validset:
        return True
    return False

def ckpt2keras(ckpt,h5fn):
    reader = tf.compat.v1.train.NewCheckpointReader(ckpt)
    f = h5py.File(h5fn,'w')
    t_g = None

    for key in sorted(reader.get_variable_to_shape_map()):
        print(key,reader.get_tensor(key).shape)
        # # 权重名称根据自己网络名称进行修改
        # if(isKeyValid(key)):
        #     keySplits = key.split(r'/')
        #     if(len(keySplits)>3):
        #         tmpconv =  keySplits[0]+'/'+keySplits[1]+'/'+keySplits[2]
        #         if(tmpconv in ckpt2kerasdict):
        #             respath = ckpt2kerasdict[tmpconv]
        #             for i in range(3,len(keySplits)):
        #                 respath += '/' + keySplits[i]
                    
        #             f[respath] = reader.get_tensor(key)
        #     else:
        #         f[key] = reader.get_tensor(key)
        # if(key.endswith('w') or key.endswith('biases')):
        #     keySplits = key.split(r'/')
        #     keyDict = keySplits[1] + '/' + keySplits
        #     f[keyDict] = reader.get_tensor(key)

def read_graph_from_pb(tf_model_path ,input_names,output_name):  
    with open(tf_model_path, 'rb') as f:
        serialized = f.read() 
    tf.reset_default_graph()
    gdef = tf.GraphDef()
    gdef.ParseFromString(serialized) 
    with tf.Graph().as_default() as g:
        tf.import_graph_def(gdef, name='') 
    
    with tf.Session(graph=g) as sess: 
        OPS=get_ops_from_pb(g,input_names,output_name)
    return OPS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts ckpt weights to deepdish hdf5")
    parser.add_argument("infile", type=str,
                        help="Path to the ckpt.")
    parser.add_argument("outfile", type=str, nargs='?', default='',
                        help="Output file (inferred if missing).")
    args = parser.parse_args()
    if args.outfile == '':
        args.outfile = os.path.splitext(args.infile)[0] + '.h5'
    outdir = os.path.dirname(args.outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    ckpt2keras(args.infile,args.outfile)
    # weights = read_ckpt(args.infile)
    # dd.io.save(args.outfile, weights)
    # weights2 = dd.io.load(args.outfile)