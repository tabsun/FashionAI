import itertools
import logging
import math
from collections import namedtuple

import time
import cv2
import numpy as np
from skimage import measure
import tensorflow as tf
from scipy.ndimage import maximum_filter, gaussian_filter

import common

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

csv_seq = list(['nkl','nkr','cf','shl','shr','apl','apr','wll','wlr','cli','clo','cri','cro','thl','thr','wbl','wbr','hll','hlr','cr','bli','blo','bri','bro'])
joint_idx_dict = dict()
joint_idx_dict['blouse'] = list(['nkl','nkr','shl','shr','cf','apl','apr','thl','thr','cli','clo','cri','cro'])
joint_idx_dict['dress'] = list(['nkl','nkr','shl','shr','cf','apl','apr','wll','wlr','cli','clo','cri','cro','hll','hlr'])
joint_idx_dict['skirt'] = list(['wbl','wbr','hll','hlr'])
joint_idx_dict['trousers'] = list(['wbl','wbr','cr','bli','blo','bri','bro'])
joint_idx_dict['outwear'] = list(['nkl','nkr','shl','shr','apl','apr','wll','wlr','cli','clo','cri','cro','thl','thr'])

# flip_idx_dict = dict()
# flip_idx_dict['blouse'] = list([1,0,3,2,-1,5,4,7,6,-1,-1,-1,-1])
# flip_idx_dict['dress'] = list([1,0,3,2,-1,5,4,7,6,-1,-1,-1,-1,14,13])
# flip_idx_dict['skirt'] = list(['wbl','wbr','hll','hlr'])
# flip_idx_dict['trousers'] = list(['wbl','wbr','cr','bli','blo','bri','bro'])
# flip_idx_dict['outwear'] = list(['nkl','nkr','shl','shr','apl','apr','wll','wlr','cli','clo','cri','cro','thl','thr'])

anchor_parts_dict = dict()
anchor_parts_dict['blouse'] = list(['nkl','nkr','shl','shr','cf','apl','apr','thl','thr'])
anchor_parts_dict['dress'] = list(['nkl','nkr','shl','shr','cf','apl','apr','wll','wlr','hll','hlr'])
anchor_parts_dict['skirt'] = list(['wbl','wbr','hll','hlr'])
anchor_parts_dict['trousers'] = list(['wbl','wbr','cr','bli','blo','bri','bro'])
anchor_parts_dict['outwear'] = list(['nkl','nkr','shl','shr','apl','apr','wll','wlr','thl','thr'])

anchor_shapes_dict = dict()
anchor_shapes_dict['dress'] = np.array( [220.06841, 81.308624, 292.07489, 81.120567, 194.30931, 99.461403,312.55582, 101.92442,261.04477, 116.4954,198.06177, 165.06046,312.92355, 165.46048,196.98837, 252.56976,318.05951, 249.82527,171.24814, 437.70828,345.02551, 438.00754]).reshape((-1,2))
anchor_shapes_dict['blouse'] = np.array([214.50197, 102.13043, 299.85519, 101.97043,169.7345, 129.9335,341.93405, 132.70079,259.3241, 138.66049,177.66478, 218.90312,335.6445, 219.74237,176.25584, 373.25601,328.59961, 372.75113]).reshape((-1,2))
anchor_shapes_dict['outwear'] = np.array([223.32913, 91.620049,288.93878, 91.409271,174.54388, 119.74562,338.12762, 119.79435,182.22247, 196.49028,329.44745, 196.13432,175.77811, 317.20087,338.26186, 316.96002,174.39052, 431.92212,342.37817, 431.07941]).reshape((-1,2))
anchor_shapes_dict['skirt'] = np.array([190.14108, 122.28519,322.81079, 122.89632,145.00015, 394.284,369.30725, 394.29788]).reshape((-1,2))
anchor_shapes_dict['trousers'] = np.array([190.70341, 121.04039,322.57724, 121.23666,256.05881, 251.42365,238.15755, 419.62891,162.99037, 409.71802,274.52762, 420.33789,350.30103, 410.48596]).reshape((-1,2))

link_idx_dict = dict()
link_idx_dict['blouse'] = list(zip(
    [1, 1, 2, 5, 5, 3,   4, 11, 13, 10, 12, 6, 7, 8],
    [2, 3, 4, 1, 2, 11, 13, 10, 12,  6,  7, 8, 9, 9]
))
link_idx_dict['dress'] = list(zip(
    [1, 1, 2, 3,  4, 11, 13, 10, 12, 6, 7, 8,  9, 14, 5, 5],
    [2, 3, 4, 11, 13,10, 12, 6,   7, 8, 9, 14,15, 15, 1, 2]
))
link_idx_dict['skirt'] = list(zip(
    [1, 1, 2, 3, 1],
    [2, 3, 4, 4, 4]
))
link_idx_dict['trousers'] = list(zip(
    [1, 1, 2, 5, 7, 4, 6, 1],
    [2, 5, 7, 4, 6, 3, 3, 3]
))
link_idx_dict['outwear'] = list(zip(
    [1, 1, 2, 3, 4, 10, 12, 9, 11, 5, 6, 7, 8, 13, 7],
    [2, 3, 4,10,12,  9, 11, 5,  6, 7, 8,13,14, 14, 8]
))

PafIDPairs = dict()
PafIDPairs['blouse'] = list(zip(
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26],
    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
))
PafIDPairs['dress'] = list(zip(
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
))
PafIDPairs['skirt'] = list(zip(
    [0, 2, 4, 6, 8],
    [1, 3, 5, 7, 9]
))
PafIDPairs['trousers'] = list(zip(
    [0, 2, 4, 6, 8, 10, 12, 14],
    [1, 3, 5, 7, 9, 11, 13, 15]
))
PafIDPairs['outwear'] = list(zip(
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
))

class ClothePart:
    def __init__(self, clothe_class, id, x, y, score):
        self.id = id
        self.part_name = joint_idx_dict[clothe_class][id]
        self.x = x
        self.y = y
        self.score = score


class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def get_sum_score(self):
        return sum([x.score for _, x in self.body_parts.items()])

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])


class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return "nkl"

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)


class PoseEstimator:
    heatmap_supress = False
    heatmap_gaussian = False
    adaptive_threshold = False

    NMS_Threshold = 0.15
    Local_PAF_Threshold = 0.2
    PAF_Count_Threshold = 5
    Part_Count_Threshold = 4
    Part_Score_Threshold = 4.5

    PartPair = namedtuple('PartPair', [
        'score',
        'part_idx1', 'part_idx2',
        'idx1', 'idx2',
        'coord1', 'coord2',
        'score1', 'score2'
    ], verbose=False)

    def __init__(self):
        pass

    @staticmethod
    def non_max_suppression(plain, window_size=3, threshold=NMS_Threshold):
        under_threshold_indices = plain < threshold
        plain[under_threshold_indices] = 0
        return plain * (plain == maximum_filter(plain, footprint=np.ones((window_size, window_size))))

    @staticmethod
    def estimate(heat_mat, paf_mat, map_cnt, clothe_class):
        if heat_mat.shape[2] == map_cnt:
            heat_mat = np.rollaxis(heat_mat, 2, 0)
        if paf_mat.shape[2] == map_cnt*2:
            paf_mat = np.rollaxis(paf_mat, 2, 0)

        if PoseEstimator.heatmap_supress:
            heat_mat = heat_mat - heat_mat.min(axis=1).min(axis=1).reshape(map_cnt, 1, 1)
            heat_mat = heat_mat - heat_mat.min(axis=2).reshape(map_cnt, heat_mat.shape[1], 1)

        if PoseEstimator.heatmap_gaussian:
            heat_mat = gaussian_filter(heat_mat, sigma=0.5)

        if PoseEstimator.adaptive_threshold:
            _NMS_Threshold = max(np.average(heat_mat) * 4.0, PoseEstimator.NMS_Threshold)
            _NMS_Threshold = min(_NMS_Threshold, 0.3)
        else:
            _NMS_Threshold = PoseEstimator.NMS_Threshold

        # extract interesting coordinates using NMS.
        coords = []     # [[coords in plane1], [....], ...]
        for plain in heat_mat[:-1]:
            nms = PoseEstimator.non_max_suppression(plain, 5, _NMS_Threshold)
            coords.append(np.where(nms >= _NMS_Threshold))

        # score pairs
        pairs_by_conn = list()
        for (part_idx1, part_idx2), (paf_x_idx, paf_y_idx) in zip(link_idx_dict[clothe_class], PafIDPairs[clothe_class]):
            part_idx1 -= 1
            part_idx2 -= 1
            pairs = PoseEstimator.score_pairs(
                part_idx1, part_idx2,
                coords[part_idx1], coords[part_idx2],
                paf_mat[paf_x_idx], paf_mat[paf_y_idx],
                heatmap=heat_mat,
                rescale=(1.0 / heat_mat.shape[2], 1.0 / heat_mat.shape[1])
            )

            pairs_by_conn.extend(pairs)

        # merge pairs to human
        # pairs_by_conn is sorted by CocoPairs(part importance) and Score between Parts.
        humans = [Human([pair]) for pair in pairs_by_conn]
        while True:
            merge_items = None
            for k1, k2 in itertools.combinations(humans, 2):
                if k1 == k2:
                    continue
                if k1.is_connected(k2):
                    merge_items = (k1, k2)
                    break

            if merge_items is not None:
                merge_items[0].merge(merge_items[1])
                humans.remove(merge_items[1])
            else:
                break

        max_score = -100000.0
        best_human = None
        for human in humans:
            if human.get_sum_score() > max_score:
                max_score = max(max_score, human.get_sum_score())
                best_human = human
        return best_human

        # reject by subset count
        # humans = [human for human in humans if human.part_count() >= PoseEstimator.PAF_Count_Threshold]

        # reject by subset max score
        # humans = [human for human in humans if human.get_max_score() >= PoseEstimator.Part_Score_Threshold]

        return humans

    @staticmethod
    def score_pairs(part_idx1, part_idx2, coord_list1, coord_list2, paf_mat_x, paf_mat_y, heatmap, rescale=(1.0, 1.0)):
        connection_temp = []

        cnt = 0
        for idx1, (y1, x1) in enumerate(zip(coord_list1[0], coord_list1[1])):
            for idx2, (y2, x2) in enumerate(zip(coord_list2[0], coord_list2[1])):
                score, count = PoseEstimator.get_score(x1, y1, x2, y2, paf_mat_x, paf_mat_y)
                cnt += 1
                if count < PoseEstimator.PAF_Count_Threshold or score <= 0.0:
                    continue
                connection_temp.append(PoseEstimator.PartPair(
                    score=score,
                    part_idx1=part_idx1, part_idx2=part_idx2,
                    idx1=idx1, idx2=idx2,
                    coord1=(x1 * rescale[0], y1 * rescale[1]),
                    coord2=(x2 * rescale[0], y2 * rescale[1]),
                    score1=heatmap[part_idx1][y1][x1],
                    score2=heatmap[part_idx2][y2][x2],
                ))

        connection = []
        used_idx1, used_idx2 = set(), set()
        for candidate in sorted(connection_temp, key=lambda x: x.score, reverse=True):
            # check not connected
            if candidate.idx1 in used_idx1 or candidate.idx2 in used_idx2:
                continue
            connection.append(candidate)
            used_idx1.add(candidate.idx1)
            used_idx2.add(candidate.idx2)

        return connection

    @staticmethod
    def get_score(x1, y1, x2, y2, paf_mat_x, paf_mat_y):
        __num_inter = 10
        __num_inter_f = float(__num_inter)
        dx, dy = x2 - x1, y2 - y1
        normVec = math.sqrt(dx ** 2 + dy ** 2)

        if normVec < 1e-4:
            return 0.0, 0

        vx, vy = dx / normVec, dy / normVec

        xs = np.arange(x1, x2, dx / __num_inter_f) if x1 != x2 else np.full((__num_inter,), x1)
        ys = np.arange(y1, y2, dy / __num_inter_f) if y1 != y2 else np.full((__num_inter,), y1)
        xs = (xs + 0.5).astype(np.int8)
        ys = (ys + 0.5).astype(np.int8)

        # without vectorization
        pafXs = np.zeros(__num_inter)
        pafYs = np.zeros(__num_inter)
        for idx, (mx, my) in enumerate(zip(xs, ys)):
            pafXs[idx] = paf_mat_x[my][mx]
            pafYs[idx] = paf_mat_y[my][mx]

        # vectorization slow?
        # pafXs = pafMatX[ys, xs]
        # pafYs = pafMatY[ys, xs]

        local_scores = pafXs * vx + pafYs * vy
        thidxs = local_scores > PoseEstimator.Local_PAF_Threshold

        return sum(local_scores * thidxs), sum(thidxs)

class TfPoseEstimator:
    ENSEMBLE = 'average'        # average, addup

    def __init__(self, graph_path, target_size=(320, 240), clothe_class='blouse'):
        self.target_size = target_size
        heat_map_cnt_dict = dict({'blouse':14, 'dress':16, 'skirt':5, 'outwear':15, 'trousers':8})
        self.heat_map_cnt = heat_map_cnt_dict[clothe_class]
        # load graph
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, name='TfPoseEstimator')
        self.persistent_sess = tf.Session(graph=self.graph)

        for op in self.graph.get_operations():
            print(op.name)

        self.tensor_image = self.graph.get_tensor_by_name('TfPoseEstimator/image:0')
        self.tensor_output = self.graph.get_tensor_by_name('TfPoseEstimator/Openpose/concat_stage7:0')

        self.heatMat = self.pafMat = None

        # warm-up
        self.persistent_sess.run(
            self.tensor_output,
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)]
            }
        )

    def __del__(self):
        self.persistent_sess.close()

    @staticmethod
    def _quantize_img(npimg):
        npimg_q = npimg + 1.0
        npimg_q /= (2.0 / 2**8)
        # npimg_q += 0.5
        npimg_q = npimg_q.astype(np.uint8)
        return npimg_q

    @staticmethod
    def draw_humans(npimg, humans, imgcopy=False):
        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        centers = {}
        for human in humans:
            # draw point
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                centers[i] = center
                cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)

            # draw line
            for pair_order, pair in enumerate(common.CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue

                npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)

        return npimg

    def _get_scaled_img(self, npimg, scale):
        get_base_scale = lambda s, w, h: max(self.target_size[0] / float(w), self.target_size[1] / float(h)) * s
        img_h, img_w = npimg.shape[:2]

        if scale is None:
            if npimg.shape[:2] != (self.target_size[1], self.target_size[0]):
                # resize
                npimg = cv2.resize(npimg, self.target_size)
            return [npimg], [(0.0, 0.0, 1.0, 1.0)]
        elif isinstance(scale, float):
            # scaling with center crop
            base_scale = get_base_scale(scale, img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale)
            ratio_x = (1. - self.target_size[0] / float(npimg.shape[1])) / 2.0
            ratio_y = (1. - self.target_size[1] / float(npimg.shape[0])) / 2.0
            roi = self._crop_roi(npimg, ratio_x, ratio_y)
            return [roi], [(ratio_x, ratio_y, 1.-ratio_x*2, 1.-ratio_y*2)]
        elif isinstance(scale, tuple) and len(scale) == 2:
            # scaling with sliding window : (scale, step)
            base_scale = get_base_scale(scale[0], img_w, img_h)
            base_scale_w = self.target_size[0] / (img_w * base_scale)
            base_scale_h = self.target_size[1] / (img_h * base_scale)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale)
            window_step = scale[1]
            rois = []
            infos = []
            for ratio_x, ratio_y in itertools.product(np.arange(0., 1.01 - base_scale_w, window_step),
                                                      np.arange(0., 1.01 - base_scale_h, window_step)):
                roi = self._crop_roi(npimg, ratio_x, ratio_y)
                rois.append(roi)
                infos.append((ratio_x, ratio_y, base_scale_w, base_scale_h))
            return rois, infos
        elif isinstance(scale, tuple) and len(scale) == 3:
            # scaling with ROI : (want_x, want_y, scale_ratio)
            base_scale = get_base_scale(scale[2], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale)
            ratio_w = self.target_size[0] / float(npimg.shape[1])
            ratio_h = self.target_size[1] / float(npimg.shape[0])

            want_x, want_y = scale[:2]
            ratio_x = want_x - ratio_w / 2.
            ratio_y = want_y - ratio_h / 2.
            ratio_x = max(ratio_x, 0.0)
            ratio_y = max(ratio_y, 0.0)
            if ratio_x + ratio_w > 1.0:
                ratio_x = 1. - ratio_w
            if ratio_y + ratio_h > 1.0:
                ratio_y = 1. - ratio_h

            roi = self._crop_roi(npimg, ratio_x, ratio_y)
            return [roi], [(ratio_x, ratio_y, ratio_w, ratio_h)]

    def _crop_roi(self, npimg, ratio_x, ratio_y):
        target_w, target_h = self.target_size
        h, w = npimg.shape[:2]
        x = max(int(w*ratio_x-.5), 0)
        y = max(int(h*ratio_y-.5), 0)
        cropped = npimg[y:y+target_h, x:x+target_w]

        cropped_h, cropped_w = cropped.shape[:2]
        if cropped_w < target_w or cropped_h < target_h:
            npblank = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)

            copy_x, copy_y = (target_w - cropped_w) // 2, (target_h - cropped_h) // 2
            npblank[copy_y:copy_y+cropped_h, copy_x:copy_x+cropped_w] = cropped
        else:
            return cropped

    def choose_the_best_pt(self, heatmap, info):
        std_size, offset_x, offset_y = info[:]
        min_val, max_val = np.min(heatmap), np.max(heatmap)
        heatmap_ = (heatmap - min_val) / (max_val - min_val) * 255.0
        left = int(offset_x * 512.0 / std_size + 0.5)
        top  = int(offset_y * 512.0 / std_size + 0.5)
        right = (512-1-left)
        bottom = (512-1-top)
        heatmap_roi = heatmap[top:bottom, left:right]

        y_arr, x_arr = np.where(heatmap_roi == np.max(heatmap_roi))
        x = x_arr[0] + left
        y = y_arr[0] + top

        if np.mean(heatmap_) > 10:
            return False,x,y
        if np.sum(heatmap_ > 128) < np.prod(heatmap.shape) * 0.02 * 0.02:
            return False,x,y
        x_arr, y_arr = np.where(heatmap_ > 128)
        x_min, x_max = np.min(x_arr), np.max(x_arr)
        y_min, y_max = np.min(y_arr), np.max(y_arr)
        if (x_max - x_min)*(y_max - y_min) > np.prod(heatmap.shape) * 0.2 * 0.2:
            return False,x,y
        return True, x, y

        # valid extraction
        labels = measure.label(heatmap_roi > heatmap_roi.mean(), connectivity=2)
        props = measure.regionprops(labels)
        max_area = 0
        best_pt = None
        for prop in props:
            if prop['area'] > max_area:
                max_area = prop['area']
                best_pt = prop['centroid']
        y, x = best_pt[:]
        y += top
        x += left
        return True, x, y
        
 
    def inference(self, input_image, clothe_class, scales=None, saveid=0):
        if input_image is None:
            raise Exception('The image is not valid. Please check your image exists.')

        # preprocess to square-size image
        ori_h, ori_w = input_image.shape[:2]
        std_size = max(ori_h, ori_w)
        whole_image = np.zeros((std_size, std_size, 3), dtype=np.float32)
        left = (std_size - ori_w) // 2
        top  = (std_size - ori_h) // 2

        whole_image[top:(top+ori_h), left:(left+ori_w), :] = input_image
        npimg = cv2.resize(whole_image, (512, 512))

        read_from_disk = False
        if not read_from_disk:       
            if not isinstance(scales, list):
                scales = [None]

            if self.tensor_image.dtype == tf.quint8:
                print "make a quantize process"
                # quantize input image
                npimg = TfPoseEstimator._quantize_img(npimg)
                pass

            t = time.time()
            rois = []
            infos = []
            for scale in scales:
                roi, info = self._get_scaled_img(npimg, scale)
                # for dubug...
                # print(roi[0].shape)
                # cv2.imshow('a', roi[0])
                # cv2.waitKey()
                rois.extend(roi)
                infos.extend(info)

            output = self.persistent_sess.run(self.tensor_output, feed_dict={self.tensor_image: rois})
            heatMats = output[:, :, :, :self.heat_map_cnt]
            pafMats = output[:, :, :, self.heat_map_cnt:]

            output_h, output_w = output.shape[1:3]
            max_ratio_w = max_ratio_h = 10000.0
            for info in infos:
                max_ratio_w = min(max_ratio_w, info[2])
                max_ratio_h = min(max_ratio_h, info[3])
            mat_w, mat_h = int(output_w/max_ratio_w), int(output_h/max_ratio_h)
            resized_heatMat = np.zeros((mat_h, mat_w, self.heat_map_cnt), dtype=np.float32)
            resized_pafMat = np.zeros((mat_h, mat_w, 2*self.heat_map_cnt), dtype=np.float32)
            resized_cntMat = np.zeros((mat_h, mat_w, 1), dtype=np.float32)
            resized_cntMat += 1e-12

            for heatMat, pafMat, info in zip(heatMats, pafMats, infos):
                w, h = int(info[2]*mat_w), int(info[3]*mat_h)
                heatMat = cv2.resize(heatMat, (w, h))
                pafMat = cv2.resize(pafMat, (w, h))
                x, y = int(info[0] * mat_w), int(info[1] * mat_h)

                if TfPoseEstimator.ENSEMBLE == 'average':
                    # average
                    resized_heatMat[max(0, y):y + h, max(0, x):x + w, :] += heatMat[max(0, -y):, max(0, -x):, :]
                    resized_pafMat[max(0,y):y+h, max(0, x):x+w, :] += pafMat[max(0, -y):, max(0, -x):, :]
                    resized_cntMat[max(0,y):y+h, max(0, x):x+w, :] += 1
                else:
                    # add up
                    resized_heatMat[max(0, y):y + h, max(0, x):x + w, :] = np.maximum(resized_heatMat[max(0, y):y + h, max(0, x):x + w, :], heatMat[max(0, -y):, max(0, -x):, :])
                    resized_pafMat[max(0,y):y+h, max(0, x):x+w, :] += pafMat[max(0, -y):, max(0, -x):, :]
                    resized_cntMat[max(0, y):y + h, max(0, x):x + w, :] += 1

            if TfPoseEstimator.ENSEMBLE == 'average':
                self.heatMat = resized_heatMat / resized_cntMat
                self.pafMat = resized_pafMat / resized_cntMat
            else:
                self.heatMat = resized_heatMat
                self.pafMat = resized_pafMat / (np.log(resized_cntMat) + 1)
            # np.save('features/heatmaps/%s/%05d.npy'%(clothe_class, saveid), self.heatMat)
            # np.save('features/pafmaps/%s/%05d.npy'%(clothe_class, saveid), self.pafMat)
            # logger.info('inference multi images: %.4f seconds.' % (time.time() - t))
        else:
            self.heatMat = np.load("features/heatmaps/%05d.npy" % saveid)
            self.pafMat  = np.load("features/pafmaps/%05d.npy" % saveid)
        ###########################################
        std_parts = []
        merge_strategy = False
        if merge_strategy:
            # Merge strategy
            human = PoseEstimator.estimate(self.heatMat, self.pafMat, self.heat_map_cnt, clothe_class)
            for i in range(len(joint_idx_dict[clothe_class])):
                if i not in human.body_parts.keys():
                    std_parts.append([-1,-1,-1])
                else:
                    body_part = human.body_parts[i]
                    one_part = [int(body_part.x * std_size + 0.5), int(body_part.y * std_size + 0.5), 1]
                    std_parts.append(one_part)
            print len(std_parts)
        else: 
            # Max strategy
            heatmaps = []
            pafmaps = []
            
            for i in range(self.heatMat.shape[2]):
                part_heatmap = self.heatMat[:,:,i]
                temp = np.zeros((part_heatmap.shape[0], part_heatmap.shape[1], 3), dtype=np.float32)
                temp[:,:,0] = part_heatmap
                temp[:,:,1] = part_heatmap
                temp[:,:,2] = part_heatmap
                resize_part_heatmap = cv2.resize(temp, dsize=tuple(npimg.shape[0:2]))
                for_save = cv2.resize(temp, dsize=(std_size, std_size))
                save_image("heatmaps/%05d_%02d.jpg" % (saveid,i), for_save[top:(top+ori_h), left:(left+ori_w), 0])
                if i != self.heatMat.shape[2]-1:
                    heatmaps.append(resize_part_heatmap[:,:,0])
            for i in range(self.pafMat.shape[2]):
                pafmap = self.pafMat[:,:,i]
                temp = np.zeros((pafmap.shape[0], pafmap.shape[1], 3), dtype=np.float32)
                temp[:,:,0] = pafmap
                temp[:,:,1] = pafmap
                temp[:,:,2] = pafmap
                resize_pafmap = cv2.resize(temp, dsize=tuple(npimg.shape[0:2]))
                for_save = cv2.resize(temp, dsize=(std_size, std_size))
                save_image("pafmaps/%05d_%02d.jpg" % (saveid,i), for_save[top:(top+ori_h), left:(left+ori_w), 0])
                pafmaps.append(resize_pafmap[:,:,0])

            for part_id, heatmap in enumerate(heatmaps):
                if(np.max(resize_part_heatmap) == np.min(resize_part_heatmap)):
                    assert("empty or all equal value heatmap detected!")
                    exit(0)

                heatmap_ = gaussian_filter(heatmap, sigma=2.0)
                valid, x, y = self.choose_the_best_pt(heatmap_, [std_size,left,top])
                # y_arr, x_arr = np.where(heatmap_ == np.max(heatmap_))
                std_parts.append([x, y, valid])
                
        # flip to get the corresponding invalid point
        # flip_mapping = flip_idx_dict[clothe_class]
        box_left = 512.0
        box_right  = 0
        box_top = 512.0
        box_bottom = 0
        for part in std_parts:
            x, y, valid = part[:]
            if valid:
                box_left  = min(x, box_left)
                box_right = max(x, box_right)
                box_top   = min(y, box_top)
                box_bottom = max(y, box_bottom)
        # for part_id, part in enumerate(std_parts):
        #     x, y, valid = part[:]
        #     if not valid and flip_mapping[part_id] != -1 and std_parts[flip_mapping[part_id]][2]:
        #         print "flipping"
        #         print std_parts[part_id]
        #         fx, fy, _ = std_parts[flip_mapping[part_id]][:]
        #         std_parts[part_id] = [box_right+box_left-fx, fy, True]
        #         print std_parts[part_id]
        # TODO waist check

        anchor_shape = anchor_shapes_dict[clothe_class]
        anchor_box_left = 512.0
        anchor_box_right  = 0
        anchor_box_top = 512.0
        anchor_box_bottom = 0
        for part in anchor_shape:
            x, y = part[:]
            anchor_box_left  = min(x, anchor_box_left)
            anchor_box_right = max(x, anchor_box_right)
            anchor_box_top   = min(y, anchor_box_top)
            anchor_box_bottom = max(y, anchor_box_bottom)

        reback_parts = []
        joint = joint_idx_dict[clothe_class]
        anchor_parts = anchor_parts_dict[clothe_class]
        for part_id,part in enumerate(std_parts):
            x, y, valid = part[:]
            part_name = joint[part_id]
            status = 1 if valid else -1
            if not valid and part_name in ['wll','wlr']:
                x, y = anchor_shape[anchor_parts.index(part_name)][:2]
                x = box_left + (x - anchor_box_left)/(anchor_box_right-anchor_box_left)*(box_right-box_left)
                y = box_top +  (y - anchor_box_top )/(anchor_box_bottom-anchor_box_top)*(box_bottom-box_top)
                status = 0

            nx = int(min(ori_w-1, max(0, float(x) / 512.0 * std_size - left + 0.5)))
            ny = int(min(ori_h-1, max(0, float(y) / 512.0 * std_size - top + 0.5)))
            reback_parts.append([nx,ny,1])
        return reback_parts

def save_image(filename, mat):
    h, w = mat.shape[0:2]
    nmat = (mat - np.min(mat)) /(np.max(mat)-np.min(mat))*255.0
    arr = np.zeros((h,w,3), dtype=np.float32)
    arr[:,:,0] = nmat
    arr[:,:,1] = nmat
    arr[:,:,2] = nmat
    cv2.imwrite(filename, arr)
