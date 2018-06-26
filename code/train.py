import matplotlib as mpl
mpl.use('Agg')      # training mode, no screen should be open. (It will block training loop)

import argparse
import logging
import os
import time


import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
# Not using multi gpus to process data
from tensorpack.dataflow.remote import RemoteDataZMQ
from tensorflow.python.ops import array_ops

from pose_dataset import get_dataflow_batch, DataFlowToQueue, FashionKeypoints
from pose_augment import set_network_input_wh, set_network_scale
from common import get_sample_images
from networks import get_network

logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def focal_loss(prediction_tensor, target_tensor, valid_tensor, weights=None, alpha=0.25, gamma=2):
    """Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    #pos_p_sub_valid = array_ops.where(valid_tensor > zeros, pos_p_sub, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    #neg_p_sub_valid = array_ops.where(valid_tensor > zeros, neg_p_sub, zeros)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(tf.multiply(per_entry_cross_ent, valid_tensor))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--model', default='mobilenet', help='model name')
    parser.add_argument('--datapath', type=str, default='/root/coco/annotations')
    parser.add_argument('--imgpath', type=str, default='/root/coco/')
    parser.add_argument('--batchsize', type=int, default=96)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max-epoch', type=int, default=30)
    parser.add_argument('--lr', type=str, default='0.01')
    parser.add_argument('--modelpath', type=str, default='/data/private/tf-openpose-models-2018-1/')
    parser.add_argument('--pretrain_basepath', type=str, default='/data/private/tf-openpose-models-2018-1/')
    parser.add_argument('--pretrain_path', type=str, default='numpy/vgg19.npy')
    parser.add_argument('--logpath', type=str, default='/data/private/tf-openpose-log-2018-1/')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--remote-data', type=str, default='', help='eg. tcp://0.0.0.0:1027')

    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    args = parser.parse_args()

    if args.gpus <= 0:
        raise Exception('gpus <= 0')

    # TODO
    # blouse      14 pts + 14 links
    # outwear     15 pts + 15 links
    # dress       16 pts + 16/17 links
    # trousers    8 pts + 7/8/9 links
    # skirt       5 pts + 4 links
    # outwear2nd  7 pts + 7 links
    map_num_dict = {"blouse":14, "outwear":15, "dress":16, "trousers":8, "skirt":5, "outwear2nd":7}
    map_num = map_num_dict[args.tag]
    # define input placeholder
    set_network_input_wh(args.input_width, args.input_height)
    scale = 4

    if args.model in ['seresnet50', 'cmu', 'vgg', 'mobilenet_thin', 'mobilenet_try', 'mobilenet_try2', 'mobilenet_try3', 'hybridnet_try']:
        scale = 8

    set_network_scale(scale)
    output_w, output_h = args.input_width // scale, args.input_height // scale

    logger.info('define model+')
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        input_node = tf.placeholder(tf.float32, shape=(args.batchsize, args.input_height, args.input_width, 3), name='image')
        # TODO
        # blouse      14 pts + 14 links
        # outwear     15 pts + 15 links
        # dress       16 pts + 16/17 links
        # trousers    8 pts + 7/8/9 links
        # skirt       5 pts + 4 links
        # outwear2nd  7 pts + 7 links
        vectmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, map_num*2), name='vectmap')
        heatmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, map_num), name='heatmap')
        offsetmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, map_num*2), name='offsetmap')
        vectmap_valid_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, map_num*2), name='vectmap_valid')
        heatmap_valid_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, map_num), name='heatmap_valid')
        # prepare data
        if not args.remote_data:
            df = get_dataflow_batch(args.datapath, args.tag, True, args.batchsize, img_path=args.imgpath)
        else:
            # transfer inputs from ZMQ
            df = RemoteDataZMQ(args.remote_data, hwm=3)
        enqueuer = DataFlowToQueue(df, [input_node, heatmap_node, vectmap_node, offsetmap_node, heatmap_valid_node, vectmap_valid_node], queue_size=1000)
        q_inp, q_heat, q_vect, q_offset, q_heat_valid, q_vect_valid = enqueuer.dequeue()

    df_valid = get_dataflow_batch(args.datapath, args.tag, False, args.batchsize, img_path=args.imgpath)
    df_valid.reset_state()
    validation_cache = []

    val_image = get_sample_images(args.input_width, args.input_height)
    logger.info('tensorboard val image: %d' % len(val_image))
    logger.info(q_inp)
    logger.info(q_heat)
    logger.info(q_vect)

    # define model for multi-gpu
    q_inp_split, q_heat_split, q_vect_split, q_offset_split, q_heat_valid_split, q_vect_valid_split = tf.split(q_inp, args.gpus), tf.split(q_heat, args.gpus), tf.split(q_vect, args.gpus), tf.split(q_offset, args.gpus), tf.split(q_heat_valid, args.gpus), tf.split(q_vect_valid, args.gpus)

    output_vectmap = []
    output_heatmap = []
    #output_offsetmap = []
    losses = []
    last_losses_l1 = []
    last_losses_l2 = []
    #last_losses_l3 = []
    outputs = []
    for gpu_id in range(args.gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                net, pretrain_path, last_layer = get_network(args.model, q_inp_split[gpu_id], clothe_class=args.tag)
                vect, heat = net.loss_last()
                #heat = net.loss_last()
                output_vectmap.append(vect)
                output_heatmap.append(heat)
                #output_offsetmap.append(offs)
                outputs.append(net.get_output())

                l1s, l2s = net.loss_l1_l2()
                #l2s = net.loss_l1_l2()
                for idx, (l1, l2) in enumerate(zip(l1s, l2s)):
                #for idx, l2 in enumerate(l2s):
                    heat_diff = tf.concat(l2,axis=0) - q_heat_split[gpu_id]
                    heat_diff_valid = tf.multiply(heat_diff, q_heat_valid_split[gpu_id])
                    vect_diff = tf.concat(l1,axis=0) - q_vect_split[gpu_id]
                    vect_diff_valid = tf.multiply(vect_diff, q_vect_valid_split[gpu_id])
                    #offset_diff = tf.concat(l3,axis=0) - q_offset_split[gpu_id]
                    #offset_diff_valid = tf.multiply(offset_diff, q_vect_valid_split[gpu_id])
                    # experiment : use focal loss to fine-tune on hard examples
                    check_shape = tf.shape(l1)
                    #loss_l1 = focal_loss(tf.concat(l1,axis=0), q_vect_split[gpu_id], q_vect_valid_split[gpu_id])
                    #loss_l2 = focal_loss(tf.concat(l2,axis=0), q_heat_split[gpu_id], q_heat_valid_split[gpu_id])
                    loss_l1 = tf.nn.l2_loss(vect_diff_valid, name='loss_l1_stage%d_tower%d' % (idx, gpu_id))
                    loss_l2 = tf.nn.l2_loss(heat_diff_valid, name='loss_l2_stage%d_tower%d' % (idx, gpu_id))
                    #loss_l3 = tf.nn.l2_loss(offset_diff_valid, name='loss_l3_stage%d_tower%d' % (idx, gpu_id))
                    # experiment : only use heatmap loss
                    losses.append(tf.reduce_mean([loss_l1, loss_l2]))
                    #losses.append(loss_l2)

                last_losses_l1.append(loss_l1)
                last_losses_l2.append(loss_l2)
                #last_losses_l3.append(loss_l3)

    outputs = tf.concat(outputs, axis=0)

    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
        # define loss
        stage2_loss = tf.reduce_sum(losses[0]) / args.batchsize 
        stage3_loss = tf.reduce_sum(losses[1]) / args.batchsize 
        stage4_loss = tf.reduce_sum(losses[2]) / args.batchsize 
        stage5_loss = tf.reduce_sum(losses[3]) / args.batchsize 


        total_loss = tf.reduce_sum(losses) / args.batchsize
        total_loss_ll_paf = tf.reduce_sum(last_losses_l1) / args.batchsize
        total_loss_ll_heat = tf.reduce_sum(last_losses_l2) / args.batchsize
        #total_loss_ll_offs = tf.reduce_sum(last_losses_l3) / args.batchsize
        total_loss_ll = tf.reduce_mean([total_loss_ll_paf, total_loss_ll_heat])

        # define optimizer
        # 121745
        step_per_epoch = 121745 // args.batchsize
        global_step = tf.Variable(0, trainable=False)
        if ',' not in args.lr:
            starter_learning_rate = float(args.lr)
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       decay_steps=30000, decay_rate=0.33, staircase=True)
        else:
            lrs = [float(x) for x in args.lr.split(',')]
            boundaries = [step_per_epoch * 5 * i for i in range(len(lrs)) if i > 0]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, lrs)

    # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.0005, momentum=0.9, epsilon=1e-10)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, global_step, colocate_gradients_with_ops=True)
    logger.info('define model-')

    # define summary
    tf.summary.scalar("s2_loss", stage2_loss)
    tf.summary.scalar("s3_loss", stage3_loss)
    tf.summary.scalar("s4_loss", stage4_loss)
    tf.summary.scalar("s5_loss", stage5_loss)

    tf.summary.scalar("loss", total_loss)
    tf.summary.scalar("loss_lastlayer", total_loss_ll)
    tf.summary.scalar("loss_lastlayer_paf", total_loss_ll_paf)
    tf.summary.scalar("loss_lastlayer_heat", total_loss_ll_heat)
    #tf.summary.scalar("loss_lastlayer_offset", total_loss_ll_offs)
    tf.summary.scalar("queue_size", enqueuer.size())
    merged_summary_op = tf.summary.merge_all()

    valid_loss = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll_paf = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll_heat = tf.placeholder(tf.float32, shape=[])
    #valid_loss_ll_offs = tf.placeholder(tf.float32, shape=[])
    sample_train = tf.placeholder(tf.float32, shape=(4, 640, 640, 3))
    sample_valid = tf.placeholder(tf.float32, shape=(12, 640, 640, 3))
    train_img = tf.summary.image('training sample', sample_train, 4)
    valid_img = tf.summary.image('validation sample', sample_valid, 12)
    valid_loss_t = tf.summary.scalar("loss_valid", valid_loss)
    valid_loss_ll_t = tf.summary.scalar("loss_valid_lastlayer", valid_loss_ll)
    merged_validate_op = tf.summary.merge([train_img, valid_img, valid_loss_t, valid_loss_ll_t])

    saver = tf.train.Saver(max_to_keep=100)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        training_name = '{}_batch:{}_lr:{}_gpus:{}_{}x{}_{}'.format(
            args.model,
            args.batchsize,
            args.lr,
            args.gpus,
            args.input_width, args.input_height,
            args.tag
        )
        logger.info('model weights initialization')
        sess.run(tf.global_variables_initializer())

        if args.checkpoint:
            logger.info('Restore from checkpoint...')
            # loader = tf.train.Saver(net.restorable_variables())
            # loader.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            # TODO
            saver = tf.train.import_meta_graph(args.checkpoint+'.meta')
            saver.restore(sess, args.checkpoint)
            #saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            logger.info('Restore from checkpoint...Done')
        elif args.pretrain_path:
            logger.info('Restore pretrained weights from %s ...' % os.path.join(args.pretrain_basepath, args.pretrain_path))
            if '.npy' in pretrain_path:
                net.load(os.path.join(args.pretrain_basepath, args.pretrain_path), sess, False)
            logger.info('Restore pretrained weights...Done')
        elif pretrain_path:
            logger.info('Restore pretrained weights...')
            if '.ckpt' in pretrain_path:
                loader = tf.train.Saver(net.restorable_variables())
                loader.restore(sess, pretrain_path)
            elif '.npy' in pretrain_path:
                net.load(os.path.join(args.pretrain_basepath, pretrain_path), sess, False)
            logger.info('Restore pretrained weights...Done')

        logger.info('prepare file writer')
        file_writer = tf.summary.FileWriter(args.logpath + training_name, sess.graph)

        logger.info('prepare coordinator')
        coord = tf.train.Coordinator()
        enqueuer.set_coordinator(coord)
        enqueuer.start()

        logger.info('Training Started.')
        time_started = time.time()
        last_gs_num = last_gs_num2 = 0
        initial_gs_num = sess.run(global_step)
        print global_step

        while True:
            _, gs_num = sess.run([train_op, global_step])

            if gs_num > step_per_epoch * args.max_epoch:
                break

            if gs_num - last_gs_num >= 100:
                check_shape_show, s2_loss, s3_loss, s4_loss, s5_loss, train_loss, train_loss_ll, train_loss_ll_paf, train_loss_ll_heat, lr_val, summary, queue_size = sess.run([check_shape, stage2_loss,stage3_loss,stage4_loss,stage5_loss,total_loss, total_loss_ll, total_loss_ll_paf, total_loss_ll_heat, learning_rate, merged_summary_op, enqueuer.size()])
                #s2_loss, s3_loss, s4_loss, s5_loss, train_loss, train_loss_ll, train_loss_ll_heat, lr_val, summary, queue_size = sess.run([stage2_loss,stage3_loss,stage4_loss,stage5_loss,total_loss, total_loss_ll, total_loss_ll_heat, learning_rate, merged_summary_op, enqueuer.size()])

                # log of training loss / accuracy
                batch_per_sec = (gs_num - initial_gs_num) / (time.time() - time_started)
                logger.info('epoch=%.2f step=%d, %0.4f examples/sec lr=%f, loss=%g, loss_ll=%g, loss_ll_paf=%g, loss_ll_heat=%g, q=%d' % (gs_num / step_per_epoch, gs_num, batch_per_sec * args.batchsize, lr_val, train_loss, train_loss_ll, train_loss_ll_paf, train_loss_ll_heat, queue_size))
                #logger.info('stage2=%g, stage3=%g, stage4=%g, stage5=%g, last=%g' % (s2_loss, s3_loss, s4_loss, s5_loss, total_loss_ll))
                last_gs_num = gs_num

                file_writer.add_summary(summary, gs_num)

            if gs_num - last_gs_num2 >= 1000:
                # save weights
                saver.save(sess, os.path.join(args.modelpath, training_name, 'model'), global_step=global_step)

                # average_loss = average_loss_ll = average_loss_ll_paf = average_loss_ll_heat = 0
                # #average_loss = average_loss_ll = average_loss_ll_heat = 0
                # total_cnt = 0

                # if len(validation_cache) == 0:
                #     for images_test, heatmaps, vectmaps, offsetmaps, _, _ in tqdm(df_valid.get_data()):
                #         validation_cache.append((images_test, heatmaps, vectmaps))
                #     df_valid.reset_state()
                #     del df_valid
                #     df_valid = None

                # # log of test accuracy
                # for images_test, heatmaps, vectmaps in validation_cache:
                #      lss, lss_ll, lss_ll_paf, lss_ll_heat, vectmap_sample, heatmap_sample = sess.run(
                #          [total_loss, total_loss_ll, total_loss_ll_paf, total_loss_ll_heat, output_vectmap, output_heatmap],
                #          feed_dict={q_inp: images_test, q_vect: vectmaps, q_heat: heatmaps}
                #      )
                #      average_loss += lss * len(images_test)
                #      average_loss_ll += lss_ll * len(images_test)
                #      average_loss_ll_paf += lss_ll_paf * len(images_test)
                #      average_loss_ll_heat += lss_ll_heat * len(images_test)
                #      #average_loss_ll_offs += lss_ll_offs * len(images_test)
                #      total_cnt += len(images_test)

                # logger.info('validation(%d) %s loss=%f, loss_ll=%f, loss_ll_paf=%f, loss_ll_heat=%f' % (total_cnt, training_name, average_loss / total_cnt, average_loss_ll / total_cnt, average_loss_ll_paf / total_cnt, average_loss_ll_heat / total_cnt))
                last_gs_num2 = gs_num

                #sample_image = [enqueuer.last_dp[0][i] for i in range(4)]
                #outputMat = sess.run(
                #    outputs,
                #    feed_dict={q_inp: np.array((sample_image + val_image)*(args.batchsize // 16))}
                #)
                #
                #heatMat = outputMat[:, :, :, :map_num]
                #pafMat = outputMat[:, :, :, map_num:]

                #sample_results = []
                #for i in range(len(sample_image)):
                #    # TODO display image
                #    test_result = FashionKeypoints.display_image(sample_image[i], heatMat[i], pafMat[i], as_numpy=True)
                #    test_result = cv2.resize(test_result, (640, 640))
                #    test_result = test_result.reshape([640, 640, 3]).astype(float)
                #    sample_results.append(test_result)

                #test_results = []
                #for i in range(len(val_image)):
                #    test_result = FashionKeypoints.display_image(val_image[i], heatMat[len(sample_image) + i], pafMat[len(sample_image) + i], as_numpy=True)
                #    test_result = cv2.resize(test_result, (640, 640))
                #    test_result = test_result.reshape([640, 640, 3]).astype(float)
                #    test_results.append(test_result)

                ## save summary
                #summary = sess.run(merged_validate_op, feed_dict={
                #    valid_loss: average_loss / total_cnt,
                #    valid_loss_ll: average_loss_ll / total_cnt,
                #    valid_loss_ll_paf: average_loss_ll_paf / total_cnt,
                #    valid_loss_ll_heat: average_loss_ll_heat / total_cnt,
                #    sample_valid: test_results,
                #    sample_train: sample_results
                #})
                #file_writer.add_summary(summary, gs_num)

        saver.save(sess, os.path.join(args.modelpath, training_name, 'model'), global_step=global_step)
    logger.info('optimization finished. %f' % (time.time() - time_started))
