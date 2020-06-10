# import tensorflow as tf
# graph = tf.get_default_graph()
# graphdef = graph.as_graph_def()

# _ = tf.train.import_meta_graph("./pretrained_weights/ssd_mobilenet_v3_small_coco_2019_08_14/model.ckpt.meta")
# summary_write = tf.summary.FileWriter("./vis_graph" , graph)


import tensorflow as tf
from tensorflow.python.framework import graph_util
import tensorflow.contrib.tensorrt as trt

# tf.reset_default_graph()  # 重置计算图
tf.compat.v1.reset_default_graph()
output_graph_path = './pretrained_weights/ssd_mobilenet_v3_small_coco_2020_01_14/covert_from_tf_small0503/frozen_inference_graph.pb'
# with tf.Session() as sess:
with tf.compat.v1.Session() as sess:
    # tf.global_variables_initializer().run()
    tf.compat.v1.global_variables_initializer().run()
    
    output_graph_def = tf.compat.v1.GraphDef() # tf.GraphDef()
    # 获得默认的图
    graph = tf.compat.v1.get_default_graph() # tf.get_default_graph()
    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")
        # 得到当前图有几个操作节点
        print("%d ops in the final graph." % len(output_graph_def.node))

        tensor_name = [tensor.name for tensor in output_graph_def.node]
        print(tensor_name)
        print('---------------------------')
        # 在log_graph文件夹下生产日志文件，可以在tensorboard中可视化模型
        summaryWriter = tf.summary.FileWriter('vis_graph/small_0503', graph)

        for op in graph.get_operations():
            # print出tensor的name和值
            print(op.name, op.values())