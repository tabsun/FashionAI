import argparse
import logging

import tensorflow as tf

from networks import get_network

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True


if __name__ == '__main__':
    """
    Use this script to just save graph and checkpoint.
    While training, checkpoints are saved. You can test them with this python code.
    """
    parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet / mobilenet_thin')
    parser.add_argument('--modelpath', type=str, default='trained/vgg_batch:96_lr:0.0005_gpus:8_368x368_blouse/model-38041')
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    input_node = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='image')

    with tf.Session(config=config) as sess:
        net, _, last_layer = get_network(args.model, input_node, sess, clothe_class=args.tag, trainable=False, model_path=args.modelpath)

        tf.train.write_graph(sess.graph_def, './tmp', 'graph.pb', as_text=True)

        graph = tf.get_default_graph()
        dir(graph)
        for n in tf.get_default_graph().as_graph_def().node:
            if 'concat_stage' not in n.name:
                continue
            print(n.name)

        saver = tf.train.Saver(max_to_keep=100)
        saver.save(sess, './tmp/chk', global_step=1)
