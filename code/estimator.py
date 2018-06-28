import itertools
import logging
import math
from collections import namedtuple

import time
import cv2
import random
import string
import numpy as np
import nudged
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
joint_idx_dict['outwear2nd'] = list(['apl','wll','thl','thr','wlr','apr'])

flip_idx_dict = dict()
flip_idx_dict['blouse'] = list(['nkr','nkl','shr','shl','cf','apr','apl','thr','thl','cri','cro','cli','clo'])
flip_idx_dict['dress'] = list(['nkr','nkl','shr','shl','cf','apr','apl','wlr','wll','cri','cro','cli','clo','hlr','hll'])
flip_idx_dict['skirt'] = list(['wbr','wbl','hlr','hll'])
flip_idx_dict['trousers'] = list(['wbr','wbl','cr','bri','bro','bli','blo'])
flip_idx_dict['outwear'] = list(['nkr','nkl','shr','shl','apr','apl','wlr','wll','cri','cro','cli','clo','thr','thl'])
flip_idx_dict['outwear2nd'] = list(['apr','wlr','thr','thl','wll','apl'])

anchor_parts_dict = dict()
anchor_parts_dict['blouse'] = list(['nkl','nkr','shl','shr','cf','apl','apr','thl','thr'])
anchor_parts_dict['dress'] = list(['nkl','nkr','shl','shr','cf','apl','apr','wll','wlr','hll','hlr'])
anchor_parts_dict['skirt'] = list(['wbl','wbr','hll','hlr'])
anchor_parts_dict['trousers'] = list(['wbl','wbr','cr','bli','blo','bri','bro'])
anchor_parts_dict['outwear'] = list(['nkl','nkr','shl','shr','apl','apr','wll','wlr','thl','thr'])
anchor_parts_dict['outwear2nd'] = list(['apl','apr','wll','wlr','thl','thr'])

anchor_shapes_dict = dict()
anchor_shapes_dict['dress'] = np.array([223.73276, 92.286179,288.45825, 91.640205,188.40132, 108.39988,323.72513, 107.55513,256.20508, 121.01356,200.65891, 166.4212, 312.19019, 165.64348, 208.00107, 228.50282,305.12192, 227.59793,171.00595, 440.29034,344.85931, 439.59186]).reshape((-1,2))
anchor_shapes_dict['blouse'] = np.array([214.50197, 102.13043, 299.85519, 101.97043,169.7345, 129.9335,341.93405, 132.70079,259.3241, 138.66049,177.66478, 218.90312,335.6445, 219.74237,176.25584, 373.25601,328.59961, 372.75113]).reshape((-1,2))
anchor_shapes_dict['outwear'] = np.array([223.32913, 91.620049,288.93878, 91.409271,174.54388, 119.74562,338.12762, 119.79435,182.22247, 196.49028,329.44745, 196.13432,175.77811, 317.20087,338.26186, 316.96002,174.39052, 431.92212,342.37817, 431.07941]).reshape((-1,2))
anchor_shapes_dict['outwear2nd'] = np.array([182.22247, 196.49028,329.44745, 196.13432,175.77811, 317.20087,338.26186, 316.96002,174.39052, 431.92212,342.37817, 431.07941]).reshape((-1,2))
anchor_shapes_dict['skirt'] = np.array([190.14108, 122.28519,322.81079, 122.89632,145.00015, 394.284,369.30725, 394.29788]).reshape((-1,2))
anchor_shapes_dict['trousers'] = np.array([190.70341, 121.04039,322.57724, 121.23666,256.05881, 251.42365,238.15755, 419.62891,162.99037, 409.71802,274.52762, 420.33789,350.30103, 410.48596]).reshape((-1,2))

mean_shape_dict = dict()
mean_shape_dict['dress'] = np.array([223.73276, 92.286179, 288.45825, 91.640205,188.40132, 108.39988,323.72513, 107.55513,256.20508, 121.01356,200.65891, 166.4212,312.19019, 165.64348,208.00107, 228.50282,305.12192, 227.59793,192.98531, 238.74632,165.67305, 232.27354,319.79297, 236.00873,347.16061, 229.78394,171.00595, 440.29034,344.85931, 439.59186]).reshape((-1,2))
mean_shape_dict['outwear2nd'] = np.array([195.75681, 202.23529,200.50102, 258.80228,170.31796, 455.51953,345.48291, 454.66263,313.16525, 257.69366,  317.49136, 200.90042]).reshape((-1,2))
mean_shape_dict['outwear'] = np.array([229.30954, 107.40636,282.88199, 106.79018,183.9771, 134.70969,328.53467, 133.63571,195.75681, 202.23529,317.49136, 200.90042,200.50102, 258.80228,313.16525, 257.69366,186.81633, 310.40048,159.33783, 314.42212,325.46011, 307.84897,352.92972, 311.9191,170.31796, 455.51953,345.48291, 454.66263]).reshape((-1,2))
mean_shape_dict['blouse'] = np.array([217.80452, 108.22989,296.90619, 108.57013,168.16328, 133.79669,346.01923, 135.48341,256.70221, 139.43452,178.60619, 214.50958,332.57919, 215.82713,174.8512, 365.88498,331.88165, 368.31155,171.58109, 335.21707,138.50085, 331.564,334.54868, 333.96722,368.10202, 331.93378]).reshape((-1,2))
mean_shape_dict['skirt'] = np.array([193.39671, 127.7915,319.78375, 127.47765,149.28661, 389.22263,364.79175, 389.27106]).reshape((-1,2))
mean_shape_dict['trousers'] = np.array([195.25809, 135.50316,318.00748, 134.94058,256.08328, 255.75751,241.24394, 410.47034,169.94127, 402.30383,271.55695, 411.44116,343.22473, 403.45416]).reshape((-1,2))

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
link_idx_dict['outwear2nd'] = list(zip(
    [1,2,3,4,5,6,2],
    [2,3,4,5,6,1,5]
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

def get_best_dist(x, y, part_name):
    if part_name in ['hlr']:
        return x #+ 0.1*y
    if part_name in ['hll']:
        return -x # + 0.1*y
    if part_name in ['wbr']:
        return x # - 0.1*y
    if part_name in ['wbl']:
        return -x #- 0.1*y

class TfPoseEstimator:
    ENSEMBLE = 'addup'        # average, addup

    def __init__(self, graph_path, target_size=(320, 240), clothe_class='blouse'):
        self.target_size = target_size
        heat_map_cnt_dict = dict({'blouse':14, 'dress':16, 'skirt':5, 'outwear':15, 'trousers':8,'outwear2nd':7})
        self.heat_map_cnt = heat_map_cnt_dict[clothe_class]
        # load graph
        tag_str = ''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(8))
        print graph_path
        print tag_str

        index = graph_path.split('_')[-1].split('.')[0]
        if index == 'graph':
            index = 0
        else:
            index = int(index)
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, name='TfPoseEstimator_%s_%s'%(clothe_class,tag_str))
        self.persistent_sess = tf.Session(graph=self.graph)

        for op in self.graph.get_operations():
            print(op.name)

        self.tensor_image = self.graph.get_tensor_by_name('TfPoseEstimator_%s_%s/image:0'%(clothe_class,tag_str))
        self.tensor_output = self.graph.get_tensor_by_name('TfPoseEstimator_%s_%s/Openpose/concat_stage7:0'%(clothe_class,tag_str))

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
                #roi_mask  = cv2.resize(mask, self.target_size)
            return [npimg], [(0.0, 0.0, 1.0, 1.0)]
        elif isinstance(scale, float):
            # scaling with center crop
            base_scale = get_base_scale(scale, img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale)
            #mask  = cv2.resize(mask, dsize=None, fx=base_scale, fy=base_scale)
            ratio_x = (1. - self.target_size[0] / float(npimg.shape[1])) / 2.0
            ratio_y = (1. - self.target_size[1] / float(npimg.shape[0])) / 2.0
            roi = self._crop_roi(npimg, ratio_x, ratio_y)
            #roi_mask = self._crop_roi(mask, ratio_x, ratio_y)
            return [roi], [(ratio_x, ratio_y, 1.-ratio_x*2, 1.-ratio_y*2)]
        elif isinstance(scale, tuple) and len(scale) == 2:
            # scaling with sliding window : (scale, step)
            base_scale = get_base_scale(scale[0], img_w, img_h)
            base_scale_w = self.target_size[0] / (img_w * base_scale)
            base_scale_h = self.target_size[1] / (img_h * base_scale)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale)
            #mask  = cv2.resize(mask, dsize=None, fx=base_scale, fy=base_scale)
            window_step = scale[1]
            rois = []
            infos = []
            #roi_masks = []
            for ratio_x, ratio_y in itertools.product(np.arange(0., 1.01 - base_scale_w, window_step),
                                                      np.arange(0., 1.01 - base_scale_h, window_step)):
                roi = self._crop_roi(npimg, ratio_x, ratio_y)
                #roi_mask = self._crop_roi(mask, ratio_x, ratio_y)
                rois.append(roi)
                #roi_masks.append(roi_mask)
                infos.append((ratio_x, ratio_y, base_scale_w, base_scale_h))
            return rois, infos
        elif isinstance(scale, tuple) and len(scale) == 3:
            # scaling with ROI : (want_x, want_y, scale_ratio)
            base_scale = get_base_scale(scale[2], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale)
            #mask = cv2.resize(mask, dsize=None, fx=base_scale, fy=base_scale)
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
            #roi_mask = self._crop_roi(mask, ratio_x, ratio_y)
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

    def sigmoid(self, x, derivative=False):
        return x * (1-x) if derivative else 1.0 / (1.0 + np.exp(-x))

    def choose_the_best_pt(self, heatmap, info, part_name):
        std_size, offset_x, offset_y = info[:]
        resize_size = heatmap.shape[0]
        min_val, max_val = np.min(heatmap), np.max(heatmap)
        heatmap_ = (heatmap - min_val) / (max_val - min_val) * 255.0
        left = int(offset_x * resize_size * 1.0 / std_size + 0.5)
        top  = int(offset_y * resize_size * 1.0 / std_size + 0.5)
        right = (resize_size-left-1)
        bottom = (resize_size-top-1)
        heatmap_roi = heatmap[top:bottom, left:right]

        y_arr, x_arr = np.where(heatmap_roi == np.max(heatmap_roi))
        x = x_arr[0] + left
        y = y_arr[0] + top
        if part_name in ['apl','apr','wll','wlr','cr'] and (x == left or y == top):
            return False, x, y
        # y_arr, x_arr = np.where(heatmap == np.max(heatmap))
        # x = x_arr[0]
        # y = y_arr[0]

        if np.mean(heatmap_) > 10:
            return False,x,y
        #if np.sum(heatmap_ > 128) < np.prod(heatmap.shape) * 0.02 * 0.02:
        if np.sum(heatmap_ > np.mean(heatmap_)) < np.prod(heatmap.shape) * 0.02 * 0.02:
            return False,x,y
        #x_arr, y_arr = np.where(heatmap_ > 128)
        #x_min, x_max = np.min(x_arr), np.max(x_arr)
        #y_min, y_max = np.min(y_arr), np.max(y_arr)
        # 3.2 x 3.2 - 8.7 x 8.7
        #if (x_max - x_min)*(y_max - y_min) > np.prod(heatmap.shape) * 0.2 * 0.2:
        #    return False,x,y
        return True, x, y

        # valid extraction
        #labels = measure.label(heatmap_roi > heatmap_roi.mean(), connectivity=2)
        #props = measure.regionprops(labels)
        #best_dist = -10000
        #best_pt = None
        #
        #if len(props) > 1:
        #    areas = [prop['area'] if prop['label'] != 0 else 0 for prop in props]
        #    areas = sorted(areas, reverse=True)
        #    area_thresh = max(areas[min(len(areas)-1, 1)], 50)
        #    for prop in props:
        #        if prop['area'] >= area_thresh:
        #            prop_id = prop['label']
        #            region_heatmap = (labels == prop_id)*heatmap_roi
        #            y_arr, x_arr = np.where(region_heatmap == np.max(region_heatmap))
        #            y, x = y_arr[0], x_arr[0]
        #            if get_best_dist(x, y, part_name) > best_dist:
        #                best_dist = get_best_dist(x, y, part_name)
        #                best_pt = [y, x]
        #            #best_pt = prop['centroid']
        #    y, x = best_pt[:]
        #    y += top
        #    x += left
        #    return True, x, y
        #else:
        #    return True, x, y

    def inverse_transform(self,tform):
        matrix = np.array(tform.get_matrix())
        s = matrix[0,0]
        r = matrix[1,0]
        tx, ty = tform.get_translation()

        det = s**2 + r**2
        shat = s / det
        rhat = -r / det
        txhat = (-s*tx - r*ty) / det
        tyhat = (r*tx - s*ty) / det
        return nudged.Transform(shat, rhat, txhat, tyhat)

    def get_bounding_box(self, pt_array, w, h, scale_x, scale_y):
        index = np.where(pt_array[:,0] > 0) and np.where(pt_array[:,1] > 0)
        left, right = int(np.min(pt_array[index,0])), int(np.max(pt_array[index,0]))
        top,  bottom= int(np.min(pt_array[index,1])), int(np.max(pt_array[index,1]))
        width = right - left + 1
        height = bottom - top + 1
        left   -= width  * (scale_x-1.0)/2.0
        right  += width  * (scale_x-1.0)/2.0
        top    -= height * (scale_y-1.0)/2.0
        bottom += height * (scale_y-1.0)/2.0
        left = max(0, left)
        top = max(0, top)
        right = min(w-1, right)
        bottom = min(h-1, bottom)
        return int(left), int(right), int(top), int(bottom)

#    def multi_stage_inference(self, input_image, clothe_class, input_size, scales=None, saveid=0):
#        predict_pts = self.inference(input_image, clothe_class, input_size, scales, saveid)
#        predictions_1st = np.array(predict_pts).copy()
#
#        joint_shape = joint_idx_dict[clothe_class]
#        assert(len(predict_pts) == len(joint_shape))
#
#        image = input_image.copy()
#        h, w = image.shape[:2]
#        std_size = int(math.sqrt(w**2 + h**2) + 0.5)
#        full_image = np.zeros((std_size, std_size, 3), dtype=np.float32)
#        offset_x, offset_y = (std_size - w) / 2, (std_size - h) / 2
#        full_image[offset_y:(offset_y+h),offset_x:(offset_x+w),:] = image
#        # modify the ground_truth points
#        for i in range(len(predict_pts)):
#            predict_pts[i][0] += offset_x
#            predict_pts[i][1] += offset_y
#
#        # filter the anchor ground_truth points
#        anchor_parts = joint_idx_dict[clothe_class] #anchor_parts_dict[clothe_class]
#        anchor_shape = mean_shape_dict[clothe_class].copy() #anchor_shapes_dict[clothe_class].copy()
#        anchor_shape[:,0] += offset_x
#        anchor_shape[:,1] += offset_y
#
#        predict_anchor_pts = []
#        mapping_anchor_pts = []
#        for part_name, pt in zip(joint_shape, predict_pts):
#            if part_name in anchor_parts and pt[2] == 1:
#                predict_anchor_pts.append(pt[:2])
#                mapping_anchor_pts.append(anchor_shape[anchor_parts.index(part_name)])
#        predict_anchor_pts_arr = np.array(predict_anchor_pts).reshape((-1,2))
#        mapping_anchor_pts_arr = np.array(mapping_anchor_pts).reshape((-1,2))
#        assert(predict_anchor_pts_arr.shape == mapping_anchor_pts_arr.shape) 
#        
#        # align the image and points
#        tform = nudged.estimate(predict_anchor_pts_arr, mapping_anchor_pts_arr)
#        tform_array = np.asarray(tform.get_matrix())
#        
#        image_align = cv2.warpPerspective(full_image, tform_array, (1000,1000))
#
#        new_predict_pts = []
#        for part_name, pt in zip(joint_shape, predict_pts):
#            # TODO experiment
#            if part_name in joint_idx_dict[clothe_class] and pt[2] == 1:
#                x, y = tform.transform([pt[0], pt[1]])
#                new_pt = [x, y, pt[2]]
#                new_predict_pts.append(new_pt)
#        
#        # crop and resize the roi image
#        # outwear / dress   2.2 1.4 1.5
#        # blouse            1.6 1.4 1.2
#        # skirt / trousers  1.2 1.2 1.8
#        if clothe_class in ['outwear','dress']:
#            extend_scale_w = 2.0
#            extend_scale_h = 1.3
#            std_ratio      = 1.5
#        if clothe_class in ['skirt','trousers']:
#            extend_scale_w = 1.4
#            extend_scale_h = 1.4
#            std_ratio      = 1.0
#        if clothe_class == 'blouse':
#            extend_scale_w = 2.0       
#            extend_scale_h = 2.0
#            std_ratio      = 1.0
#        new_predict_pts_arr = np.array(new_predict_pts).reshape((-1,3))
#        left, right, top, bottom = self.get_bounding_box(new_predict_pts_arr, extend_scale_w, extend_scale_h)
#        scale = (bottom-top)*1.0/(right-left)
#        if scale > std_ratio:
#            # extend width
#            extend_w = min((int((bottom-top+1)/std_ratio)-(right-left))/2, min(image_align.shape[1]-right, left))
#            left -= extend_w
#            right += extend_w
#        else:
#            # extend height
#            extend_h = min((int((right-left+1)*std_ratio)-(bottom-top))/2, min(image_align.shape[0]-bottom, top))
#            top -= extend_h
#            bottom += extend_h
#        # get this image and inference for the second time
#        std_image = image_align[top:bottom, left:right, :]
#        cv2.imwrite("haha/%05d.jpg"%saveid,std_image)
#        predict_pts2 = self.inference(std_image, clothe_class, input_size, scales, saveid)
#
#        # reback to original coordinates
#        predictions_2nd = []
#        inv_tform = self.inverse_transform(tform)
#        for pt in predict_pts2:
#            inv_x, inv_y = inv_tform.transform([pt[0]+left, pt[1]+top])
#            inv_x -= offset_x
#            inv_y -= offset_y
#            predictions_2nd.append([x, y, pt[2]])
#                        
#        # merge the first and second detection result
#        points = []
#        assert(len(predictions_1st) == len(predictions_2nd))
#        for pt1, pt2 in zip(predictions_1st, predictions_2nd):
#            if pt1[2] == 1:
#                points.append([int(pt1[0]),int(pt1[1]),int(pt1[2])])
#            elif pt2[2] >= 0:
#                pt2[2] = 1
#                points.append([int(pt2[0]),int(pt2[1]),int(pt2[2])])
#            else:
#                pt2[2] = 1
#                points.append([int(pt2[0]),int(pt2[1]),int(pt2[2])])
#        return points
    def multi_stage_inference(self, input_image, clothe_class, input_size, scales=None, saveid=0):
        predict_pts = self.inference(input_image, clothe_class, input_size, scales, saveid)
        predictions_1st = np.array(predict_pts).copy()

        joint_shape = joint_idx_dict[clothe_class]
        assert(len(predict_pts) == len(joint_shape))

        image = input_image.copy()
        h, w = image.shape[:2]
        
        # crop and resize the roi image
        # outwear / dress   2.2 1.4 1.5
        # blouse            1.6 1.4 1.2
        # skirt / trousers  1.2 1.2 1.8
        if clothe_class in ['outwear','dress']:
            extend_scale_w = 2.0
            extend_scale_h = 2.0
            std_ratio      = 1.0
        if clothe_class in ['skirt','trousers']:
            extend_scale_w = 2.0
            extend_scale_h = 2.0
            std_ratio      = 1.0
        if clothe_class == 'blouse':
            extend_scale_w = 2.0       
            extend_scale_h = 2.0
            std_ratio      = 1.0
        predict_pts_arr = np.array(predictions_1st).reshape((-1,3))
        left, right, top, bottom = self.get_bounding_box(predict_pts_arr, w, h, extend_scale_w, extend_scale_h)
        scale = (bottom-top)*1.0/(right-left)
        if scale > std_ratio:
            # extend width
            extend_w = ((bottom-top+1)/std_ratio-(right-left))/2
            left -= extend_w
            right += extend_w
            left = max(0, int(left))
            right = min(w-1, int(right))
        else:
            # extend height
            extend_h = ((right-left+1)*std_ratio-(bottom-top))/2
            top -= extend_h
            bottom += extend_h
            top = max(0, int(top))
            bottom = min(h-1, int(bottom))
        # get this image and inference for the second time
        std_image = image[top:bottom, left:right, :]
        #cv2.imwrite("haha/%05d.jpg"%saveid,std_image)
        predict_pts2 = self.inference(std_image, clothe_class, input_size, scales, saveid)

        # reback to original coordinates
        predictions_2nd = []
        for pt in predict_pts2:
            x, y, v = pt[:]
            x += left
            y += top
            predictions_2nd.append([x, y, v])
                        
        # merge the first and second detection result
        points = []
        assert(len(predictions_1st) == len(predictions_2nd))
        for pt1, pt2 in zip(predictions_1st, predictions_2nd):
            #if pt2[2] == 1:
            points.append([int(pt2[0]),int(pt2[1]),int(pt2[2])])
            #elif pt2[2] >= 0:
            #    pt2[2] = 1
            #    points.append([int(pt2[0]),int(pt2[1]),int(pt2[2])])
            #else:
            #    pt2[2] = 1
            #    points.append([int(pt2[0]),int(pt2[1]),int(pt2[2])])
        return points

    def extract_heatmap(self, image, clothe_class, scales=None, saveid=0):
        read_from_disk = False
        if not read_from_disk:       
            if not isinstance(scales, list):
                scales = [None]

            if self.tensor_image.dtype == tf.quint8:
                print "make a quantize process"
                # quantize input image
                image = TfPoseEstimator._quantize_img(image)
                pass

            t = time.time()
            rois = []
            infos = []
            for scale in scales:
                roi, info = self._get_scaled_img(image, scale)
                rois.extend(roi)
                infos.extend(info)
            #for index, (roi, roi_mask) in enumerate(zip(rois, roi_masks)):
            #    cv2.imwrite('%04d_roi.jpg' % index, roi)
            #    cv2.imwrite('%04d_mask.jpg' % index, roi_mask*(roi_mask>0.5)*255)
            output = self.persistent_sess.run(self.tensor_output, feed_dict={self.tensor_image: rois})
            heatMats = output[:, :, :, :self.heat_map_cnt]
            pafMats  = output[:, :, :, self.heat_map_cnt:]
            #heatMats = output[:, :, :, 2*self.heat_map_cnt:3*self.heat_map_cnt]
            #pafMats = output[:, :, :, :2*self.heat_map_cnt]

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

            #g_x, g_y, g_w, g_h = g_info[:]
            for heatMat_, pafMat, info in zip(heatMats, pafMats, infos):
                w, h = int(info[2]*mat_w), int(info[3]*mat_h)
                # valid region
                #temp_mask = cv2.resize(roi_mask, (heatMat_.shape[1], heatMat_.shape[0]))[:,:,0] > 0.5
                #for heatmap_id in xrange(heatMat_.shape[2]):
                #    heatMat_[:,:,heatmap_id] = heatMat_[:,:,heatmap_id] * temp_mask

                #heatMat = self.sigmoid(heatMat_)                
                heatMat = cv2.resize(heatMat_, (w, h))
                pafMat = cv2.resize(pafMat, (w, h))
                heatMat = self.sigmoid(heatMat)
 
                #off_x, off_y = int(w * g_x), int(h * g_y)
                #x, y = int((info[0]+g_x*info[2]) * mat_w), int((info[1]+g_y*info[3]) * mat_h)
                x, y = int(info[0] * mat_w), int(info[1] * mat_h)
                #w, h = int(w * g_w), int(h * g_h)
                if TfPoseEstimator.ENSEMBLE == 'average':
                    # average
                    resized_heatMat[max(0, y):y + h, max(0, x):x + w, :] += heatMat[max(0,-y):, max(0,-x):, :]
                    resized_pafMat[max(0,y):y+h, max(0, x):x+w, :] += pafMat[max(0,-y):, max(0,-x):, :]
                    resized_cntMat[max(0,y):y+h, max(0, x):x+w, :] += 1
                elif TfPoseEstimator.ENSEMBLE == 'global':
                    if info[0]==0.0 and info[1]==0.0 and info[2]==1.0 and info[3]==1.0:
                        resized_heatMat[max(0, y):y + h, max(0, x):x + w, :] += heatMat[max(0,-y):, max(0,-x):, :]
                        resized_pafMat[max(0,y):y+h, max(0, x):x+w, :] += pafMat[max(0,-y):, max(0,-x):, :]
                        resized_cntMat[max(0,y):y+h, max(0, x):x+w, :] += 1 
                else:
                    # add up
                    resized_heatMat[max(0, y):y + h, max(0, x):x + w, :] = np.maximum(resized_heatMat[max(0, y):y + h, max(0, x):x + w, :], heatMat[max(0,-y):, max(0,-x):, :])
                    resized_pafMat[max(0,y):y+h, max(0, x):x+w, :] += pafMat[max(0,-y):, max(0,-x):]
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
            self.heatMat = np.load("features/heatmaps/%s/%05d.npy" % (clothe_class,saveid))
            self.pafMat  = np.load("features/pafmaps/%s/%05d.npy" % (clothe_class,saveid))

    def extract_points2(self, heatmaps, clothe_class, input_size, info, saveid=0):
        self.heatMat = heatmaps
        return self.extract_points(clothe_class, input_size, info, saveid)

    def extract_points(self, clothe_class, input_size, info, saveid=0):
        std_size, left, top, ori_w, ori_h = info[:]
        joint = joint_idx_dict[clothe_class]
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
                resize_part_heatmap = cv2.resize(temp, dsize=tuple(input_size))
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
                resize_pafmap = cv2.resize(temp, dsize=tuple(input_size))
                for_save = cv2.resize(temp, dsize=(std_size, std_size))
                save_image("pafmaps/%05d_%02d.jpg" % (saveid,i), for_save[top:(top+ori_h), left:(left+ori_w), 0])
                pafmaps.append(resize_pafmap[:,:,0])

            for part_id, heatmap in enumerate(heatmaps):
                if(np.max(heatmap) == np.min(heatmap)):
                    print np.max(resize_part_heatmap)
                    print("empty or all equal value heatmap detected!")
                    # exit(0)
                    std_parts.append([0, 0, False, True])
                    

                part_name = joint[part_id]
                heatmap_ = gaussian_filter(heatmap, sigma=2.0)
                valid, x, y = self.choose_the_best_pt(heatmap_, [std_size,left,top], part_name)
                # y_arr, x_arr = np.where(heatmap_ == np.max(heatmap_))
                std_parts.append([x, y, valid, False])

        resize_size = input_size[0]        

        # TODO waist check
        #std_parts = self.recheck_valid(std_parts, clothe_class) 
        anchor_parts = anchor_parts_dict[clothe_class]
        anchor_shape = anchor_shapes_dict[clothe_class]
        
        src = []
        dst = []
        for pt, part_name in zip(std_parts, joint):
            if pt[2] and part_name in anchor_parts:
                src.append(anchor_shape[anchor_parts.index(part_name)])
                dst.append([pt[0],pt[1]])
        tform = nudged.estimate(src, dst)

        reback_parts = []
        mean_shape = mean_shape_dict[clothe_class]
        for part_id,part in enumerate(std_parts):
            x, y, valid, fullfill = part[:]
            part_name = joint[part_id]
            status = 1 if valid else -1
            if (not valid and ((clothe_class=='dress' and part_name in ['hll', 'hlr', 'wll','wlr']) or 
                               (clothe_class!='dress' and part_name in ['wll', 'wlr']))) or fullfill:
                x, y = mean_shape[joint.index(part_name)]
                x, y = tform.transform([x,y])
                status = 0
            #if not valid and part_name in ['cri','cro','cli','clo']:
            #    x, y, _, _ = std_parts[joint.index(part_mapping[part_name])]
            #    status = 0

            nx = int(min(ori_w-1, max(0, round(float(x) / resize_size * std_size - left))))
            ny = int(min(ori_h-1, max(0, round(float(y) / resize_size * std_size - top ))))
            if nx < 0 or ny < 0 or nx > ori_w-1 or ny > ori_h-1:
                print("How could be?")
                exit(0)
                status = -1
            reback_parts.append([nx,ny,status])
        return reback_parts

    def recheck_valid(self, parts, clothe_class):
        std_parts = []
        if clothe_class == 'skirt':
            wbl_pt, wbr_pt, hll_pt, hlr_pt = parts[:]
            
    def pose_align(self, img, init_parts, clothe_class):
        image = img.copy()
        h, w = image.shape[:2]
        std_size = int(math.sqrt(w**2 + h**2) + 0.5)
        full_image = np.zeros((std_size, std_size, 3), dtype=np.float32)
        offset_x, offset_y = (std_size - w) // 2, (std_size - h) // 2
        full_image[offset_y:(offset_y+h),offset_x:(offset_x+w),:] = image
        # modify the ground_truth points
        joint_shape = [(pt[0],pt[1]) for pt in init_parts]
        joint_shape_ = []
        for i in range(len(joint_shape)):
            x, y = joint_shape[i][:]
            if x < 0 or y < 0:
                joint_shape_.append([x, y])
            else:
                joint_shape_.append([x + offset_x, y + offset_y])
        joint_shape = joint_shape_

        # filter the anchor ground_truth points
        anchor_parts = anchor_parts_dict[clothe_class]
        joint_parts = joint_idx_dict[clothe_class]
        mean_shape = mean_shape_dict[clothe_class].copy() + 300.0 #anchor_shapes_dict[clothe_class].copy()
        #anchor_shape[:,0] += offset_x
        #anchor_shape[:,1] += offset_y

        pts = []
        mts = []
        for part_name, pt, mt in zip(joint_parts, joint_shape, mean_shape):
            if pt[0] < 0 or pt[1] < 0 or part_name not in anchor_parts:
                continue
            pts.append(list(pt))
            mts.append(list(mt))
        pts_arr = np.array(pts).reshape((-1,2))
        mts_arr = np.array(mts).reshape((-1,2))
        assert(pts_arr.shape == mts_arr.shape) 
        
        # align the image and points
        tform = nudged.estimate(pts_arr, mts_arr)
        tform_array = np.asarray(tform.get_matrix())
        
        image_align = cv2.warpPerspective(full_image, tform_array, (1500,1500))

        # transform the groundtruth points
        new_pts = []
        for pt in joint_shape:
            x, y = pt
            if x < 0 or y < 0: 
                new_pts.append([x, y])
            else:
                x, y = tform.transform(list(pt))
                new_pts.append([x,y])

        # crop and resize the roi image
        # outwear / dress   2.2 1.4 1.5
        # blouse            1.6 1.4 1.2
        # skirt / trousers  1.2 1.2 1.8
        extend_scale_w = 1.4
        extend_scale_h = 1.4
        std_ratio      = 1.0

        new_pts_arr = np.array(new_pts).reshape((-1,2))
        align_h, align_w = image_align.shape[:2]
        left, right, top, bottom = self.get_bounding_box(new_pts_arr, align_w, align_h, extend_scale_w, extend_scale_h)
        scale = (bottom-top)*1.0/(right-left)
        if scale > std_ratio:
            # extend width
            extend_w = min((int((bottom-top+1)/std_ratio)-(right-left))/2, min(align_w-right, left))
            left -= extend_w
            right += extend_w
        else:
            # extend height
            extend_h = min((int((right-left+1)*std_ratio)-(bottom-top))/2, min(align_h-bottom, top))
            top -= extend_h
            bottom += extend_h
        # get this image and inference for the second time
        std_image = image_align[top:bottom, left:right, :]

        return std_image, [(offset_x,offset_y), tform, (left,right,top,bottom)]

    def refine_inference(self, input_image, clothe_class, input_size, scales=None, saveid=0, need_heatmaps=False, init_parts=None):
        image, info = self.pose_align(input_image, init_parts, clothe_class) 
        offset, tform, box = info
        parts = self.inference(image, clothe_class, 368, scales=[1.0], saveid=saveid)
        for pt in parts:
            cv2.circle(image, tuple(pt[:2]), 4, (255,128,0), -1)
        cv2.imwrite('show_imgs/%05d.jpg'%saveid, image)
        
        inv_tform = self.inverse_transform(tform)
        left, _, top, _ = box
        joint_shape_full, joint_shape_2nd = joint_idx_dict[clothe_class], joint_idx_dict[clothe_class]
        modify_parts = init_parts
        for pt, pt_name in zip(parts, joint_shape_2nd):
            # calculate reback points' position
            x, y = pt[0] + left, pt[1] + top
            inv_x, inv_y = inv_tform.transform([x,y])
            x, y = inv_x - offset[0], inv_y - offset[1]
            index = joint_shape_full.index(pt_name)
            modify_parts[index] = [int(x), int(y), 1]
        return modify_parts

    def inference(self, input_image, clothe_class, input_size, scales=None, saveid=0, need_heatmaps=False):
        if input_image is None:
            raise Exception('The image is not valid. Please check your image exists.')

        # preprocess to square-size image
        ori_h, ori_w = input_image.shape[:2]
        std_size = max(ori_h, ori_w)
        whole_image = np.zeros((std_size, std_size, 3), dtype=np.float32)
        whole_mask  = np.zeros((std_size, std_size, 3), dtype=np.float32)
        left = (std_size - ori_w) // 2
        top  = (std_size - ori_h) // 2

        whole_image[top:(top+ori_h), left:(left+ori_w), :] = input_image
        whole_mask[ top:(top+ori_h), left:(left+ori_w), :] = 1.0
        npimg = cv2.resize(whole_image, (input_size, input_size))
        npimg_mask = cv2.resize(whole_mask, (input_size, input_size))

        joint = joint_idx_dict[clothe_class]
        #info = [left*1.0/std_size, top*1.0/std_size, ori_w*1.0/std_size, ori_h*1.0/std_size] 
        merge_points = False
        self.extract_heatmap(npimg, clothe_class, scales, saveid)
        heatmaps_ori = self.heatMat.copy()
        if merge_points:
            parts_ori = self.extract_points(clothe_class, (input_size,input_size), [std_size, left, top, ori_w, ori_h])
        #for i in range(heatmaps_ori.shape[2]):
        #    save_image('heatmap_%02d.jpg' % i, heatmaps_ori[:,:,i])
        # flip the image
        
        flip_joint = flip_idx_dict[clothe_class]

        npimg_flip = cv2.flip(npimg, 1)
        self.extract_heatmap(npimg_flip, clothe_class, scales, saveid)
        merge_heatmaps = self.heatMat.copy() #np.zeros((npimg.shape[0],npimg.shape[1],self.heatMat.shape[2]),dtype=np.float32)
        if merge_points:
            parts_flip = self.extract_points(clothe_class, (input_size,input_size), [std_size, left, top, ori_w, ori_h])

        # MERGE the flip and original heatmaps
        if not merge_points:
            for i in xrange(self.heatMat.shape[2]):
                # background heatmap
                if i == self.heatMat.shape[2]-1:
                    flip_index = i
                else:
                    flip_index = flip_joint.index(joint[i])
                flip_heatmap = self.heatMat[:,:,flip_index]
                heatmap = heatmaps_ori[:,:,i]
                merge_heatmap = np.mean(np.array([heatmap,cv2.flip(flip_heatmap,1)]), axis=0)
                merge_heatmaps[:,:,i] = merge_heatmap            
            if need_heatmaps:
                return merge_heatmaps, [std_size, left, top, ori_w, ori_h]
            self.heatMat = merge_heatmaps
            return self.extract_points(clothe_class, (input_size,input_size), [std_size, left, top, ori_w, ori_h],saveid)
        else:
            reback_parts = []
            for i in xrange(len(joint)):
                pt_orig = parts_ori[i]
                pt_flip = parts_flip[flip_joint.index(joint[i])]
                x = (pt_orig[0]+ori_w-pt_flip[0]) // 2
                y = (pt_orig[1]+pt_flip[1]) // 2
                status = max(pt_orig[2], pt_flip[2])
                reback_parts.append([x,y,status])
            return reback_parts
        

def save_image(filename, mat):
    h, w = mat.shape[0:2]
    nmat = (mat - np.min(mat)) /(np.max(mat)-np.min(mat))*255.0
    arr = np.zeros((h,w,3), dtype=np.float32)
    arr[:,:,0] = nmat
    arr[:,:,1] = nmat
    arr[:,:,2] = nmat
    cv2.imwrite(filename, arr)
