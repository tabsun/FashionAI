import logging
import math
import multiprocessing
import struct
import sys
import threading

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from contextlib import contextmanager

import os
import random
import requests
import cv2
import numpy as np
import time

import tensorflow as tf

from tensorpack.dataflow import MultiThreadMapData
from tensorpack.dataflow.image import MapDataComponent
from tensorpack.dataflow.common import BatchData, MapData
from tensorpack.dataflow import PrefetchData
from tensorpack.dataflow.base import RNGDataFlow, DataFlowTerminated

from pycocotools.coco import COCO
from pose_augment import pose_flip, pose_rotation, pose_to_img, pose_crop_random, \
    pose_resize_shortestedge_random, pose_resize_shortestedge_fixed, pose_crop_center, pose_random_scale

logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger('pose_dataset')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

mplset = False


class FashionMetadata:
    __csv_seq = list(['nkl','nkr','cf','shl','shr','apl','apr','wll','wlr','cli','clo','cri','cro','thl','thr','wbl','wbr','hll','hlr','cr','bli','blo','bri','bro'])
    # TODO
    __joint_idx_dict = dict()
    __joint_idx_dict['blouse'] = list(['nkl','nkr','shl','shr','cf','apl','apr','thl','thr','cli','clo','cri','cro'])
    __joint_idx_dict['dress'] = list(['nkl','nkr','shl','shr','cf','apl','apr','wll','wlr','cli','clo','cri','cro','hll','hlr'])
    __joint_idx_dict['skirt'] = list(['wbl','wbr','hll','hlr'])
    __joint_idx_dict['trousers'] = list(['wbl','wbr','cr','bli','blo','bri','bro'])
    __joint_idx_dict['outwear'] = list(['nkl','nkr','shl','shr','apl','apr','wll','wlr','cli','clo','cri','cro','thl','thr'])

    __link_idx_dict = dict()
    # TODO error in links design
    __link_idx_dict['blouse'] = list(zip(
        [1, 1, 2, 5, 5, 3,   4, 11, 13, 10, 12, 6, 7, 8],
        [2, 3, 4, 1, 2, 11, 13, 10, 12,  6,  7, 8, 9, 9]
    ))
    __link_idx_dict['dress'] = list(zip(
        [1, 1, 2, 3,  4, 11, 13, 10, 12, 6, 7, 8,  9, 14, 5, 5],
        [2, 3, 4, 11, 13,10, 12, 6,   7, 8, 9, 14,15, 15, 1, 2]
    ))
    __link_idx_dict['skirt'] = list(zip(
        [1, 1, 2, 3, 1],
        [2, 3, 4, 4, 4]
    ))
    __link_idx_dict['trousers'] = list(zip(
        [1, 1, 2, 5, 7, 4, 6, 1],
        [2, 5, 7, 4, 6, 3, 3, 3]
    ))   
    __link_idx_dict['outwear'] = list(zip(
        [1, 1, 2, 3, 4, 10, 12, 9, 11, 5, 6, 7, 8, 13, 7],
        [2, 3, 4,10,12,  9, 11, 5,  6, 7, 8,13,14, 14, 8]
    ))

    @staticmethod
    def parse_float(four_np):
        assert len(four_np) == 4
        return struct.unpack('<f', bytes(four_np))[0]

    @staticmethod
    def parse_floats(four_nps, adjust=0):
        assert len(four_nps) % 4 == 0
        return [(FashionMetadata.parse_float(four_nps[x*4:x*4+4]) + adjust) for x in range(len(four_nps) // 4)]

    def __init__(self, idx, img_url, annotations, clothe_class, sigma):
        self.idx = idx
        self.img_url = img_url
        self.img = None
        self.sigma = sigma

        # Image width & height will be read in read_image_url.
        #self.height = int(img_meta['height'])
        #self.width = int(img_meta['width'])
        
        # meta.joint_list:
        # [ [(x1,y1), (x2,y2),.....,(x18,y18)] 
        #   [(x1,y1), (x2,y2),.....,(x18,y18)]
        #   [(x1,y1), (x2,y2),.....,(x18,y18)]
        #   ...                                ]
        joint_filter_str = FashionMetadata.__joint_idx_dict[clothe_class] if clothe_class in FashionMetadata.__joint_idx_dict.keys() else []
        new_joint = []
        for part_name in joint_filter_str:
            x, y, v = annotations[FashionMetadata.__csv_seq.index(part_name)]
            if(v >= 1):
                new_joint.append((float(x),float(y)))
            else:
                new_joint.append((-1000.,-1000.))
            
        assert(len(new_joint) == len(joint_filter_str))
        self.joint_list = []
        self.joint_list.append(new_joint)
        self.clothe_class = clothe_class
        self.kp_num = len(joint_filter_str)
        # logger.debug('joint size=%d' % len(self.joint_list))

    def get_heatmap(self, target_size):
        heatmap = np.zeros((self.kp_num+1, self.height, self.width), dtype=np.float32)

        for joints in self.joint_list:
            for idx, point in enumerate(joints):
                if point[0] < 0 or point[1] < 0:
                    continue
                FashionMetadata.put_heatmap(heatmap, idx, point, self.sigma)

        heatmap = heatmap.transpose((1, 2, 0))

        # background
        heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)

        if target_size:
            heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_AREA)

        return heatmap.astype(np.float16)

    @staticmethod
    def put_heatmap(heatmap, plane_idx, center, sigma):
        center_x, center_y = center
        _, height, width = heatmap.shape[:3]

        th = 4.6052
        delta = math.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))

        x1 = int(min(width, center_x + delta * sigma))
        y1 = int(min(height, center_y + delta * sigma))

        for y in range(y0, y1):
            for x in range(x0, x1):
                d = (x - center_x) ** 2 + (y - center_y) ** 2
                exp = d / 2.0 / sigma / sigma
                if exp > th:
                    continue
                heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
                heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)

    def get_vectormap(self, target_size):
        vectormap = np.zeros(((self.kp_num+1)*2, self.height, self.width), dtype=np.float32)
        countmap = np.zeros((self.kp_num+1, self.height, self.width), dtype=np.int16)
        link_vec = FashionMetadata.__link_idx_dict[self.clothe_class]

        for joints in self.joint_list:
            for plane_idx, (j_idx1, j_idx2) in enumerate(link_vec):
                j_idx1 -= 1
                j_idx2 -= 1

                center_from = joints[j_idx1]
                center_to = joints[j_idx2]

                if center_from[0] < -100 or center_from[1] < -100 or center_to[0] < -100 or center_to[1] < -100:
                    continue

                FashionMetadata.put_vectormap(vectormap, countmap, plane_idx, center_from, center_to)

        vectormap = vectormap.transpose((1, 2, 0))
        nonzeros = np.nonzero(countmap)
        for p, y, x in zip(nonzeros[0], nonzeros[1], nonzeros[2]):
            if countmap[p][y][x] <= 0:
                continue
            vectormap[y][x][p*2+0] /= countmap[p][y][x]
            vectormap[y][x][p*2+1] /= countmap[p][y][x]

        if target_size:
            vectormap = cv2.resize(vectormap, target_size, interpolation=cv2.INTER_AREA)

        return vectormap.astype(np.float16)

    @staticmethod
    def put_vectormap(vectormap, countmap, plane_idx, center_from, center_to, threshold=8):
        _, height, width = vectormap.shape[:3]

        vec_x = center_to[0] - center_from[0]
        vec_y = center_to[1] - center_from[1]

        min_x = max(0, int(min(center_from[0], center_to[0]) - threshold))
        min_y = max(0, int(min(center_from[1], center_to[1]) - threshold))

        max_x = min(width, int(max(center_from[0], center_to[0]) + threshold))
        max_y = min(height, int(max(center_from[1], center_to[1]) + threshold))

        norm = math.sqrt(vec_x ** 2 + vec_y ** 2)
        if norm == 0:
            return

        vec_x /= norm
        vec_y /= norm

        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                bec_x = x - center_from[0]
                bec_y = y - center_from[1]
                dist = abs(bec_x * vec_y - bec_y * vec_x)

                if dist > threshold:
                    continue

                countmap[plane_idx][y][x] += 1

                # TODO : whether here it should be += vec_x or vec_y ?
                vectormap[plane_idx*2+0][y][x] = +vec_x
                vectormap[plane_idx*2+1][y][x] = +vec_y

class Fashion:
    def __init__(self, file_path, clothe_class):
        self.image_paths = []
        self.annotations = []
        with open(file_path,'r') as fin:
            lines = fin.readlines()[1:]
            for line in lines:
                elems = line.strip().split(',')
                if elems[1] == clothe_class:
                    self.image_paths.append(elems[0])
                    points = np.empty([0,3])
                    for i in xrange(2,26):
                        point = np.array(elems[i].split('_')).reshape(1,3)
                        points = np.vstack((points, point))
                    self.annotations.append(points)
        assert(len(self.image_paths) == len(self.annotations))
    def size(self):
        return len(self.image_paths)
         
class FashionKeypoints(RNGDataFlow):
    @staticmethod
    def display_image(inp, heatmap, vectmap, as_numpy=False):
        global mplset
        # if as_numpy and not mplset:
        #     import matplotlib as mpl
        #     mpl.use('Agg')
        mplset = True
        import matplotlib.pyplot as plt

        fig = plt.figure()
        a = fig.add_subplot(2, 2, 1)
        a.set_title('Image')
        plt.imshow(FashionKeypoints.get_bgimg(inp))

        a = fig.add_subplot(2, 2, 2)
        a.set_title('Heatmap')
        plt.imshow(FashionKeypoints.get_bgimg(inp, target_size=(heatmap.shape[1], heatmap.shape[0])), alpha=0.5)
        tmp = np.amax(heatmap, axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        tmp2 = vectmap.transpose((2, 0, 1))
        tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
        tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

        a = fig.add_subplot(2, 2, 3)
        a.set_title('Vectormap-x')
        plt.imshow(FashionKeypoints.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        a = fig.add_subplot(2, 2, 4)
        a.set_title('Vectormap-y')
        plt.imshow(FashionKeypoints.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        if not as_numpy:
            plt.show()
        else:
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            fig.clear()
            plt.close()
            return data
    @staticmethod
    def get_bgimg(inp, target_size=None):
        inp = cv2.cvtColor(inp.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if target_size:
            inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)
        return inp

    def __init__(self, path, img_path=None, is_train=True, clothe_class='', decode_img=True, only_idx=-1):
        self.is_train = is_train
        self.decode_img = decode_img
        self.only_idx = only_idx
        self.clothe_class = clothe_class
        
        if is_train:
            whole_path = os.path.join(path, 'train/train.csv')
        else:
            whole_path = os.path.join(path, 'val/val.csv')
        self.img_path = (img_path if img_path is not None else '') + ('train/' if is_train else 'val/')
        self.fashion = Fashion(whole_path, clothe_class)

        logger.info('%s dataset %d' % (whole_path, self.size()))

    def size(self):
        return self.fashion.size()

    def get_data(self):
        idxs = np.arange(self.size())
        if self.is_train:
            self.rng.shuffle(idxs)
        else:
            pass

        image_paths = self.fashion.image_paths
        annotations = self.fashion.annotations
        for idx in idxs:
            img_url = os.path.join(self.img_path, image_paths[idx])
            anns = annotations[idx]
            meta = FashionMetadata(idx, img_url, anns, self.clothe_class, sigma=8.0)

            total_keypoints = np.sum(anns[:,2] >= 1)
            if total_keypoints == 0 and random.uniform(0, 1) > 0.2:
                continue

            yield [meta]


def read_image_url(metas):
    for meta in metas:
        img_str = open(meta.img_url, 'rb').read()

        if not img_str:
            logger.warning('image not read, path=%s' % meta.img_url)
            raise Exception()

        nparr = np.fromstring(img_str, np.uint8)
        meta.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        meta.width = float(meta.img.shape[1])
        meta.height = float(meta.img.shape[0])
    return metas


def get_dataflow(path, is_train, clothe_class, img_path=None):
    ds = FashionKeypoints(path, img_path, is_train, clothe_class)       # read data from lmdb
    if is_train:
        ds = MapData(ds, read_image_url)
        ds = MapDataComponent(ds, pose_random_scale)
        ds = MapDataComponent(ds, pose_rotation)
        ds = MapDataComponent(ds, pose_flip)
        ds = MapDataComponent(ds, pose_resize_shortestedge_random)
        ds = MapDataComponent(ds, pose_crop_random)
        ds = MapData(ds, pose_to_img)
        # augs = [
        #     imgaug.RandomApplyAug(imgaug.RandomChooseAug([
        #         imgaug.GaussianBlur(max_size=3)
        #     ]), 0.7)
        # ]
        # ds = AugmentImageComponent(ds, augs)
        ds = PrefetchData(ds, 1000, multiprocessing.cpu_count() * 2)
    else:
        ds = MultiThreadMapData(ds, nr_thread=16, map_func=read_image_url, buffer_size=10)
        ds = MapDataComponent(ds, pose_resize_shortestedge_fixed)
        ds = MapDataComponent(ds, pose_crop_center)
        ds = MapData(ds, pose_to_img)
        ds = PrefetchData(ds, 100, multiprocessing.cpu_count() // 4)

    return ds


def get_dataflow_batch(path, clothe_class, is_train, batchsize, img_path=None):
    logger.info('dataflow img_path=%s' % img_path)
    ds = get_dataflow(path, is_train, clothe_class, img_path=img_path)
    ds = BatchData(ds, batchsize)
    if is_train:
        ds = PrefetchData(ds, 10, 2)
    else:
        ds = PrefetchData(ds, 50, 2)

    return ds


class DataFlowToQueue(threading.Thread):
    def __init__(self, ds, placeholders, queue_size=5):
        super(DataFlowToQueue, self).__init__()
        self.daemon = True

        self.ds = ds
        self.placeholders = placeholders
        self.queue = tf.FIFOQueue(queue_size, [ph.dtype for ph in placeholders], shapes=[ph.get_shape() for ph in placeholders])
        self.op = self.queue.enqueue(placeholders)
        self.close_op = self.queue.close(cancel_pending_enqueues=True)

        self._coord = None
        self._sess = None

        self.last_dp = None

    @contextmanager
    def default_sess(self):
        if self._sess:
            with self._sess.as_default():
                yield
        else:
            logger.warning("DataFlowToQueue {} wasn't under a default session!".format(self.name))
            yield

    def size(self):
        return self.queue.size()

    def start(self):
        self._sess = tf.get_default_session()
        super(DataFlowToQueue, self).start()

    def set_coordinator(self, coord):
        self._coord = coord

    def run(self):
        with self.default_sess():
            try:
                while not self._coord.should_stop():
                    try:
                        self.ds.reset_state()
                        while True:
                            for dp in self.ds.get_data():
                                feed = dict(zip(self.placeholders, dp))
                                self.op.run(feed_dict=feed)
                                self.last_dp = dp
                    except (tf.errors.CancelledError, tf.errors.OutOfRangeError, DataFlowTerminated):
                        logger.error('err type1, placeholders={}'.format(self.placeholders))
                        sys.exit(-1)
                    except Exception as e:
                        logger.error('err type2, err={}, placeholders={}'.format(str(e), self.placeholders))
                        if isinstance(e, RuntimeError) and 'closed Session' in str(e):
                            pass
                        else:
                            logger.exception("Exception in {}:{}".format(self.name, str(e)))
                        sys.exit(-1)
            except Exception as e:
                logger.exception("Exception in {}:{}".format(self.name, str(e)))
            finally:
                try:
                    self.close_op.run()
                except Exception:
                    pass
                logger.info("{} Exited.".format(self.name))

    def dequeue(self):
        return self.queue.dequeue()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    from src.pose_augment import set_network_input_wh
    # set_network_input_wh(368, 368)
    set_network_input_wh(480, 320)

    df = get_dataflow('/root/coco/annotations', True, img_path='http://gpu-twg.kakaocdn.net/braincloud/COCO/')
    # df = get_dataflow('/root/coco/annotations', False, img_path='http://gpu-twg.kakaocdn.net/braincloud/COCO/')

    # TestDataSpeed(df).start()
    # sys.exit(0)

    with tf.Session() as sess:
        df.reset_state()
        t1 = time.time()
        for idx, dp in enumerate(df.get_data()):
            if idx == 0:
                for d in dp:
                    logger.info('%d dp shape={}'.format(d.shape))
            print(time.time() - t1)
            t1 = time.time()
            CocoPose.display_image(dp[0], dp[1].astype(np.float32), dp[2].astype(np.float32))
            print(dp[1].shape, dp[2].shape)
            pass

    logger.info('done')
