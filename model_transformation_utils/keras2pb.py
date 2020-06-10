from keras.models import load_model
import tensorflow as tf
import os 
import os.path as osp
from keras import backend as K

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from tqdm import tqdm
import json
from imageio import imread
from object_detection.utils import label_map_util

#路径参数
input_path = 'input path'
weight_file = 'weight.h5'
weight_file_path = osp.join(input_path,weight_file)
output_graph_name = weight_file[:-3] + '.pb'



#转换函数
def h5_to_pb(h5_model,output_dir,model_name,out_prefix = "output_",log_tensorboard = True):
    if osp.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        print("h5_model.output[i]",h5_model.output[i].name)
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i],out_prefix + str(i + 1))
    K.set_learning_phase(0)
    sess = K.get_session()
    from tensorflow.python.framework import graph_util,graph_io
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)
    graph_io.write_graph(main_graph,output_dir,name = model_name,as_text = False)
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir,model_name),output_dir)
# #输出路径
# output_dir = osp.join(os.getcwd(),"trans_model")
# #加载模型
# h5_model = load_model(weight_file_path)
# h5_to_pb(h5_model,output_dir = output_dir,model_name = output_graph_name)
# print('model saved')

import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  imgdata = np.array(image.getdata())
#   print("imgdata",imgdata.shape)
  if(len(imgdata.shape)==1):
      return imgdata.reshape((im_height, im_width, 1)).astype(np.uint8).repeat(3,axis=2)
  return imgdata.reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def load_pb(pb_file_path):
    sess = tf.Session()
    with gfile.FastGFile(pb_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    return sess

def predict(pb,img_path = './imgs/test/000000551439.jpg'):
    input_x = sess.graph.get_tensor_by_name('input_1:0')

    image_size=(320,320)
    image = Image.open(img_path)
    image = image.resize((image_size[0],image_size[1]),Image.ANTIALIAS)
    image_np = load_image_into_numpy_array(image)

    input_images = []
    input_images.append(image_np)
    input_images = np.array(input_images,dtype=np.float32)

    #输出
    op = sess.graph.get_tensor_by_name('output_1:0')
    #预测结果
    ret = sess.run(op, {input_x: input_images})
    # print(ret)
    return ret


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

if __name__ == "__main__":
    sess = load_pb(pb_file_path='./ourmodels/convert_large.pb')
    confidence_threshold = 0.1
    image_size=(320,320)
    PATH_TO_LABELS = './trained_models/mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    coco_root = '/home/huangxiaoyu/datasets/coco/'

    gt_json_path = os.path.join(coco_root,"annotations","instances_val2014.json")
    coco_gt = COCO(gt_json_path)
    # image_ids = coco_gt.getImgIds()

    fn = open('mscoco_val_ids.txt','r')
    image_ids = []
    for line in fn:
        line = line.split()[0]
        image_ids.append(int(line))

    # We'll only load one image in this example.
    img_dir = os.path.join(coco_root,"images","val2014")
    results = []
    processed_image_ids = []
    for image_id in tqdm(image_ids):
        image_info = coco_gt.loadImgs(image_id)[0]

        processed_image_ids.append(image_id)

        img_path = os.path.join(img_dir, image_info['file_name'])
        # print(img_path)
        orig_images = [] # Store the images here.
        input_images = [] # Store resized versions of the images here.

        orig_images.append(imread(img_path))

        y_pred = predict(sess,img_path)

        # print("y_pred",y_pred)
        # print("len(y_pred)",len(y_pred),"len(y_pred[0])",len(y_pred[0]))

        y_pred_thresh = []
        for k in range(len(y_pred)):
            yd = y_pred[k]
            if(yd[1] >= confidence_threshold):
                y_pred_thresh.append(yd)

        print("y_pred_thresh",y_pred_thresh)

        for box in y_pred_thresh:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            xmin = box[2] * orig_images[0].shape[1] / image_size[0]
            ymin = box[3] * orig_images[0].shape[0] / image_size[1]
            xmax = box[4] * orig_images[0].shape[1] / image_size[1]
            ymax = box[5] * orig_images[0].shape[0] / image_size[0]
            
            # print("xmin:{},ymin:{},xmax:{},ymax:{}".format(xmin,ymin,xmax,ymax))

            # color = colors[int(box[0])]
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
    json.dump(results, open('./res/large_pb_preds.json', 'w'), indent=4)


    _eval(coco_gt, image_ids, './res/large_pb_preds.json')