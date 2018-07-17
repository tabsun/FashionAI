import caffe
import numpy as np
import os

def extract_caffe_model(prototxt, weights, output_file):
    net = caffe.Net(prototxt, caffe.TEST)
    net.copy_from(weights)

    data_dict = dict()
    for item in net.params.items():
        name, layer = item
        layer_param = dict()

        length = len(net.params[name][0].data.shape)
        layer_param['weights']  = np.transpose(net.params[name][0].data, tuple(reversed(range(length))))
        layer_param['biases'] = net.params[name][1].data

        data_dict[name] = layer_param

    np.save(output_file, data_dict, allow_pickle=True)
 
extract_caffe_model('./vgg19.prototxt','./vgg19.caffemodel','vgg19.npy')
data_dict = np.load('./vgg19.npy', encoding='bytes').item()

for key in data_dict.keys():
    print key
    print data_dict[key]['weights'].shape
