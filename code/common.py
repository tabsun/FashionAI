from enum import Enum

import tensorflow as tf
import cv2


regularizer_conv = 0.004
regularizer_dsconv = 0.0004
batchnorm_fused = True
activation_fn = tf.nn.relu

class BlousePart(Enum):
    nkl = 0
    nkr = 1
    shl = 2
    shr = 3
    cf = 4
    apl = 5
    apr = 6
    thl = 7
    thr = 8
    cli = 9
    clo = 10
    cri = 11
    cro = 12
    Background = 13

class DressPart(Enum):
    nkl = 0
    nkr = 1
    shl = 2
    shr = 3
    cf = 4
    apl = 5
    apr = 6
    wll = 7
    wlr = 8
    cli = 9
    clo = 10
    cri = 11
    cro = 12
    hll = 13
    hlr = 14
    Background = 15

class OutwearPart(Enum):
    nkl = 0
    nkr = 1
    shl = 2
    shr = 3
    apl = 4
    apr = 5
    wll = 6
    wlr = 7
    cli = 8
    clo = 9
    cri = 10
    cro = 11
    thl = 12
    thr = 13
    Background = 14

class SkirtPart(Enum):
    wbl = 0
    wbr = 1
    hll = 2
    hlr = 3
    Background = 4

class TrousersPart(Enum):
    wbl = 0
    wbr = 1
    cr = 2
    bli = 3
    blo = 4
    bri = 5
    bro = 6
    Background = 7

class MPIIPart(Enum):
    RAnkle = 0
    RKnee = 1
    RHip = 2
    LHip = 3
    LKnee = 4
    LAnkle = 5
    RWrist = 6
    RElbow = 7
    RShoulder = 8
    LShoulder = 9
    LElbow = 10
    LWrist = 11
    Neck = 12
    Head = 13

    @staticmethod
    def from_coco(human):
        # t = {
        #     MPIIPart.RAnkle: CocoPart.RAnkle,
        #     MPIIPart.RKnee: CocoPart.RKnee,
        #     MPIIPart.RHip: CocoPart.RHip,
        #     MPIIPart.LHip: CocoPart.LHip,
        #     MPIIPart.LKnee: CocoPart.LKnee,
        #     MPIIPart.LAnkle: CocoPart.LAnkle,
        #     MPIIPart.RWrist: CocoPart.RWrist,
        #     MPIIPart.RElbow: CocoPart.RElbow,
        #     MPIIPart.RShoulder: CocoPart.RShoulder,
        #     MPIIPart.LShoulder: CocoPart.LShoulder,
        #     MPIIPart.LElbow: CocoPart.LElbow,
        #     MPIIPart.LWrist: CocoPart.LWrist,
        #     MPIIPart.Neck: CocoPart.Neck,
        #     MPIIPart.Nose: CocoPart.Nose,
        # }

        t = [
            (MPIIPart.Head, CocoPart.Nose),
            (MPIIPart.Neck, CocoPart.Neck),
            (MPIIPart.RShoulder, CocoPart.RShoulder),
            (MPIIPart.RElbow, CocoPart.RElbow),
            (MPIIPart.RWrist, CocoPart.RWrist),
            (MPIIPart.LShoulder, CocoPart.LShoulder),
            (MPIIPart.LElbow, CocoPart.LElbow),
            (MPIIPart.LWrist, CocoPart.LWrist),
            (MPIIPart.RHip, CocoPart.RHip),
            (MPIIPart.RKnee, CocoPart.RKnee),
            (MPIIPart.RAnkle, CocoPart.RAnkle),
            (MPIIPart.LHip, CocoPart.LHip),
            (MPIIPart.LKnee, CocoPart.LKnee),
            (MPIIPart.LAnkle, CocoPart.LAnkle),
        ]

        pose_2d_mpii = []
        visibilty = []
        for mpi, coco in t:
            if coco.value not in human.body_parts.keys():
                pose_2d_mpii.append((0, 0))
                visibilty.append(False)
                continue
            pose_2d_mpii.append((human.body_parts[coco.value].x, human.body_parts[coco.value].y))
            visibilty.append(True)
        return pose_2d_mpii, visibilty

CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]   # = 19
CocoPairsRender = CocoPairs[:-2]
CocoPairsNetwork = [
    (12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1), (2, 3), (4, 5),
    (6, 7), (8, 9), (10, 11), (28, 29), (30, 31), (34, 35), (32, 33), (36, 37), (18, 19), (26, 27)
 ]  # = 19

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def read_imgfile(path, width, height):
    val_image = cv2.imread(path, cv2.IMREAD_COLOR)
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    return val_image


def get_sample_images(w, h):
    val_image = [
        read_imgfile('/home/shy/projects/tf-openpose/images/p1.jpg', w, h),
        read_imgfile('/home/shy/projects/tf-openpose/images/p2.jpg', w, h),
        read_imgfile('/home/shy/projects/tf-openpose/images/p3.jpg', w, h),
        read_imgfile('/home/shy/projects/tf-openpose/images/golf.jpg', w, h),
        read_imgfile('/home/shy/projects/tf-openpose/images/hand1.jpg', w, h),
        read_imgfile('/home/shy/projects/tf-openpose/images/hand2.jpg', w, h),
        read_imgfile('/home/shy/projects/tf-openpose/images/apink1_crop.jpg', w, h),
        read_imgfile('/home/shy/projects/tf-openpose/images/ski.jpg', w, h),
        read_imgfile('/home/shy/projects/tf-openpose/images/apink2.jpg', w, h),
        read_imgfile('/home/shy/projects/tf-openpose/images/apink3.jpg', w, h),
        read_imgfile('/home/shy/projects/tf-openpose/images/handsup1.jpg', w, h),
        read_imgfile('/home/shy/projects/tf-openpose/images/p3_dance.png', w, h),
    ]
    return val_image
