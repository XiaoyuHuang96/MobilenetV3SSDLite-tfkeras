import h5py


def print_keras_wegiths(weight_file_path):
    f = h5py.File(weight_file_path,'r')  # 读取weights h5文件返回File类
    try:
        print(len(f.keys()))
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))  # 输出储存在File类中的attrs信息，一般是各层的名称

        for layer, g in f.items():  # 读取各层的名称以及包含层信息的Group类
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.items(): # 输出储存在Group类中的attrs信息，一般是各层的weights和bias及他们的名称
                print("1-{}: {}".format(key, value))  
                print("list {} including:".format(key))
                for k,v in value.items():
                    print("2-{}: {}".format(k, v))

                    if(hasattr(v,"items")):
                        for kk,vv in v.items():
                            print("3-{}:{}".format(kk,vv))

                            if(hasattr(vv,"items")):
                                for kkk,vvv in vv.items():
                                    print("4-{}:{}".format(kkk,vvv))

                                    if(hasattr(vvv,"items")):
                                        for kkkk,vvvv in vvv.items():
                                            print("5-{}:{}".format(kkkk,vvvv))

                                            if(hasattr(vvvv,"items")):
                                                for k5,v5 in vvvv.items():
                                                    print("6-{}:{}".format(k5,v5))

                print("list {} end:".format(key))  
                    # if(hasattr())
                    # for ik,iv in v.items():
                    #     print("      {}: {}".format(ik, iv))  

            # print("    Dataset:")
            # for name, d in g.items(): # 读取各层储存具体信息的Dataset类
            #     print("      {}: {}".format(name, d.value.shape)) # 输出储存在Dataset中的层名称和权重，也可以打印dataset的attrs，但是keras中是空的
            #     print("      {}: {}".format(name. d.value))
    finally:
        f.close()

# print_keras_wegiths('./pretrained_weights/ssd_mobilenet_v3_large_coco_2019_08_14/model.h5')
print_keras_wegiths('./pretrained_weights/tf_convert_mbv3ssdlite_small_0518.h5')