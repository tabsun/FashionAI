import math
import random

import cv2
import numpy as np
import nudged
from tensorpack.dataflow.imgaug.geometry import RotationAndCropValid

_network_w = 368
_network_h = 368
_scale = 2

joint_idx_dict = dict()
joint_idx_dict['blouse'] = list(['nkl','nkr','shl','shr','cf','apl','apr','thl','thr','cli','clo','cri','cro'])
joint_idx_dict['dress'] = list(['nkl','nkr','shl','shr','cf','apl','apr','wll','wlr','cli','clo','cri','cro','hll','hlr'])
joint_idx_dict['skirt'] = list(['wbl','wbr','hll','hlr'])
joint_idx_dict['trousers'] = list(['wbl','wbr','cr','bli','blo','bri','bro'])
joint_idx_dict['outwear'] = list(['nkl','nkr','shl','shr','apl','apr','wll','wlr','cli','clo','cri','cro','thl','thr'])
joint_idx_dict['outwear2nd'] = list(['apl','wll','thl','thr','wlr','apr'])

anchor_parts_dict = dict()
anchor_parts_dict['blouse'] = list(['nkl','nkr','shl','shr','cf','apl','apr','thl','thr'])
anchor_parts_dict['dress'] = list(['nkl','nkr','shl','shr','cf','apl','apr','wll','wlr','hll','hlr'])
anchor_parts_dict['skirt'] = list(['wbl','wbr','hll','hlr'])
anchor_parts_dict['trousers'] = list(['wbl','wbr','cr','bli','blo','bri','bro'])
anchor_parts_dict['outwear'] = list(['nkl','nkr','shl','shr','apl','apr','wll','wlr','thl','thr'])
anchor_parts_dict['outwear2nd'] = list(['apl','apr','wll','wlr','thl','thr'])

anchor_shape_dict = dict()
anchor_shape_dict['dress'] = np.array([223.73276, 92.286179, 288.45825, 91.640205,188.40132, 108.39988,323.72513, 107.55513,256.20508, 121.01356,200.65891, 166.4212,312.19019, 165.64348,208.00107, 228.50282,305.12192, 227.59793,192.98531, 238.74632,165.67305, 232.27354,319.79297, 236.00873,347.16061, 229.78394,171.00595, 440.29034,344.85931, 439.59186]).reshape((-1,2))
anchor_shape_dict['outwear'] = np.array([229.30954, 107.40636,282.88199, 106.79018,183.9771, 134.70969,328.53467, 133.63571,195.75681, 202.23529,317.49136, 200.90042,200.50102, 258.80228,313.16525, 257.69366,186.81633, 310.40048,159.33783, 314.42212,325.46011, 307.84897,352.92972, 311.9191,170.31796, 455.51953,345.48291, 454.66263]).reshape((-1,2))
anchor_shape_dict['blouse'] = np.array([217.80452, 108.22989,296.90619, 108.57013,168.16328, 133.79669,346.01923, 135.48341,256.70221, 139.43452,178.60619, 214.50958,332.57919, 215.82713,174.8512, 365.88498,331.88165, 368.31155,171.58109, 335.21707,138.50085, 331.564,334.54868, 333.96722,368.10202, 331.93378]).reshape((-1,2))
anchor_shape_dict['skirt'] = np.array([193.39671, 127.7915,319.78375, 127.47765,149.28661, 389.22263,364.79175, 389.27106]).reshape((-1,2))
anchor_shape_dict['trousers'] = np.array([195.25809, 135.50316,318.00748, 134.94058,256.08328, 255.75751,241.24394, 410.47034,169.94127, 402.30383,271.55695, 411.44116,343.22473, 403.45416]).reshape((-1,2))
anchor_shape_dict['outwear2nd'] = np.array([195.75681, 202.23529,200.50102, 258.80228,170.31796, 455.51953,345.48291, 454.66263,313.16525, 257.69366,  317.49136, 200.90042]).reshape((-1,2))

def set_network_input_wh(w, h):
    global _network_w, _network_h
    _network_w, _network_h = w, h


def set_network_scale(scale):
    global _scale
    _scale = scale

def get_bounding_box(pt_array, w, h, scale_x, scale_y):
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

def pose_align(meta):
    if len(meta.joint_list) != 1:
        print "each image can only caontain one object."
        exit(0)

    image = meta.img.copy()
    h, w = image.shape[:2]
    std_size = int(math.sqrt(w**2 + h**2) + 0.5)
    full_image = np.zeros((std_size, std_size, 3), dtype=np.float32)
    offset_x, offset_y = (std_size - w) // 2, (std_size - h) // 2
    full_image[offset_y:(offset_y+h),offset_x:(offset_x+w),:] = image
    # modify the ground_truth points
    joint_shape = meta.joint_list[0]
    joint_shape_ = []
    for i in range(len(joint_shape)):
        x, y = joint_shape[i][:]
        if x < 0 or y < 0:
            joint_shape_.append([x, y])
        else:
            joint_shape_.append([x + offset_x, y + offset_y])
    joint_shape = joint_shape_

    # filter the anchor ground_truth points
    anchor_parts = anchor_parts_dict[meta.clothe_class]
    mean_shape = anchor_shape_dict[meta.clothe_class].copy() + 300.0 #anchor_shapes_dict[clothe_class].copy()
    #anchor_shape[:,0] += offset_x
    #anchor_shape[:,1] += offset_y

    pts = []
    mts = []
    joint_parts = joint_idx_dict[meta.clothe_class]
    anchor_parts = anchor_parts_dict[meta.clothe_class]
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
    left, right, top, bottom = get_bounding_box(new_pts_arr, align_w, align_h, extend_scale_w, extend_scale_h)
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
    new_pts_tuple = []
    for index, pt in enumerate(new_pts):
        if new_pts[index][0] < 0 or new_pts[index][1] < 0:
            new_pts_tuple.append(tuple(new_pts[index]))
        else:
            new_pts_tuple.append( (int(new_pts[index][0] - left), int(new_pts[index][1] - top)) )

    meta.img = std_image
    meta.width = std_image.shape[1]
    meta.height = std_image.shape[0]
    meta.joint_list = [new_pts_tuple]
    # show
    # show_img = meta.img.copy()
    # name_str = ""
    # for joint in meta.joint_list:
    #     for pt in joint:
    #         if pt[0] > 0 and pt[1] > 0:
    #             cv2.circle(show_img, tuple(pt), 3, (255,0,0), -1)
    #             name_str += "%d_%d_"%(pt[0], pt[1])
    # cv2.imwrite('show_imgs/%s.jpg'%name_str, show_img)
    return meta

def pose_random_scale(meta):
    scalew = random.uniform(0.8, 1.2)
    scaleh = random.uniform(0.8, 1.2)
    neww = int(meta.width * scalew)
    newh = int(meta.height * scaleh)
    dst = cv2.resize(meta.img, (neww, newh), interpolation=cv2.INTER_AREA)

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            # if point[0] <= 0 or point[1] <= 0 or int(point[0] * scalew + 0.5) > neww or int(
            #                         point[1] * scaleh + 0.5) > newh:
            #     adjust_joint.append((-1, -1))
            #     continue
            adjust_joint.append((int(point[0] * scalew + 0.5), int(point[1] * scaleh + 0.5)))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = float(neww), float(newh)
    meta.img = dst
    return meta


def pose_resize_shortestedge_fixed(meta):
    ratio_w = _network_w / float(meta.width)
    ratio_h = _network_h / float(meta.height)
    ratio = max(ratio_w, ratio_h)
    return pose_resize_shortestedge(meta, int(min(meta.width * ratio + 0.5, meta.height * ratio + 0.5)))


def pose_resize_shortestedge_random(meta):
    ratio_w = _network_w / float(meta.width)
    ratio_h = _network_h / float(meta.height)
    ratio = min(ratio_w, ratio_h)
    target_size = int(min(meta.width * ratio + 0.5, meta.height * ratio + 0.5))
    target_size = int(target_size * random.uniform(0.95, 1.6))
    # target_size = int(min(_network_w, _network_h) * random.uniform(0.7, 1.5))
    return pose_resize_shortestedge(meta, target_size)


def pose_resize_shortestedge(meta, target_size):
    global _network_w, _network_h
    img = meta.img
    
    # adjust image
    scale = float(target_size) / min(meta.height, meta.width)
    if meta.height < meta.width:
        newh, neww = target_size, int(scale * meta.width + 0.5)
    else:
        newh, neww = int(scale * meta.height + 0.5), target_size
    dst = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)

    pw = ph = 0
    if neww < _network_w or newh < _network_h:
        pw = max(0, (_network_w - neww) // 2)
        ph = max(0, (_network_h - newh) // 2)
        mw = (_network_w - neww) % 2
        mh = (_network_h - newh) % 2
        color = random.randint(0, 255)
        dst = cv2.copyMakeBorder(dst, ph, ph+mh, pw, pw+mw, cv2.BORDER_CONSTANT, value=(color, 0, 0))

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            # if point[0] <= 0 or point[1] <= 0 or int(point[0]*scale+0.5) > neww or int(point[1]*scale+0.5) > newh:
            #     adjust_joint.append((-1, -1))
            #     continue
            adjust_joint.append((int(point[0]*scale+0.5) + pw, int(point[1]*scale+0.5) + ph))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = neww + pw * 2.0, newh + ph * 2.0
    meta.img = dst
    return meta


    

def pose_crop_center(meta):
    global _network_w, _network_h
    target_size = (_network_w, _network_h)
    x = (int(meta.width) - target_size[0]) // 2 if meta.width > target_size[0] else 0
    y = (int(meta.height) - target_size[1]) // 2 if meta.height > target_size[1] else 0

    return pose_crop(meta, x, y, target_size[0], target_size[1])


def pose_crop_random(meta):
    global _network_w, _network_h
    target_size = (_network_w, _network_h)

    for _ in range(50):
        x = random.randrange(0, int(meta.width) - target_size[0]) if meta.width > target_size[0] else 0
        y = random.randrange(0, int(meta.height) - target_size[1]) if meta.height > target_size[1] else 0

        # TODO
        # check whether any face is inside the box to generate a reasonably-balanced datasets
        #for joint in meta.joint_list:
        #    if x <= joint[CocoPart.Nose.value][0] < x + target_size[0] and y <= joint[CocoPart.Nose.value][1] < y + target_size[1]:
        #        break

    return pose_crop(meta, x, y, target_size[0], target_size[1])


def pose_crop(meta, x, y, w, h):
    # adjust image
    target_size = (w, h)

    img = meta.img
    resized = img[y:y+target_size[1], x:x+target_size[0], :]
    ori_w, ori_h, ori_x, ori_y = w, h, x, y
    ori_joint_list = meta.joint_list
    ori_shape = meta.img.shape
    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            # if point[0] <= 0 or point[1] <= 0:
            #     adjust_joint.append((-1000, -1000))
            #     continue
            new_x, new_y = point[0] - x, point[1] - y
            # if new_x <= 0 or new_y <= 0 or new_x > target_size[0] or new_y > target_size[1]:
            #     adjust_joint.append((-1, -1))
            #     continue
            adjust_joint.append((new_x, new_y))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = target_size
    meta.img = resized
    return meta


def pose_flip(meta):
    r = random.uniform(0, 1.0)
    if r > 0.5:
        return meta

    img = meta.img
    img = cv2.flip(img, 1)

    ori_joint_list = meta.joint_list
    # flip meta
    # TODO No background
    flip_relation_dict = dict()
    flip_relation_dict['blouse'] = [1,0,3,2,4,6,5,8,7,11,12,9,10]
    flip_relation_dict['outwear'] = [1,0,3,2,5,4,7,6,10,11,8,9,13,12]
    flip_relation_dict['trousers'] = [1,0,2,5,6,3,4]
    flip_relation_dict['skirt'] = [1,0,3,2]
    flip_relation_dict['dress'] = [1,0,3,2,4,6,5,8,7,11,12,9,10,14,13]
    flip_relation_dict['outwear2nd'] = [5,4,3,2,1,0]
    flip_relation = flip_relation_dict[meta.clothe_class]

    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for kp_part in flip_relation:
            point = joint[kp_part]
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            # if point[0] <= 0 or point[1] <= 0:
            #     adjust_joint.append((-1, -1))
            #     continue
            adjust_joint.append((meta.width - point[0], point[1]))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list

    meta.img = img
    return meta


def pose_rotation(meta):
    deg = random.uniform(-15.0, 15.0)
    img = meta.img

    center = (img.shape[1] * 0.5, img.shape[0] * 0.5)       # x, y
    rot_m = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), deg, 1)
    ret = cv2.warpAffine(img, rot_m, img.shape[1::-1], flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
    if img.ndim == 3 and ret.ndim == 2:
        ret = ret[:, :, np.newaxis]
    neww, newh = RotationAndCropValid.largest_rotated_rect(ret.shape[1], ret.shape[0], deg)
    neww = min(neww, ret.shape[1])
    newh = min(newh, ret.shape[0])
    newx = int(center[0] - neww * 0.5)
    newy = int(center[1] - newh * 0.5)
    img = ret[newy:newy + newh, newx:newx + neww]

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            # if point[0] <= 0 or point[1] <= 0:
            #     adjust_joint.append((-1, -1))
            #     continue
            x, y = _rotate_coord((meta.width, meta.height), (newx, newy), point, deg)
            adjust_joint.append((x, y))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = neww, newh
    meta.img = img
    return meta


def _rotate_coord(shape, newxy, point, angle):
    angle = -1 * angle / 180.0 * math.pi

    ox, oy = shape
    px, py = point

    ox /= 2
    oy /= 2

    qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    new_x, new_y = newxy

    qx += ox - new_x
    qy += oy - new_y

    return int(qx + 0.5), int(qy + 0.5)


def pose_to_img(meta_l):
    global _network_w, _network_h, _scale
    # TODO reback to 2nd class
    #meta_l[0].filter_parts()
    return [
        meta_l[0].img.astype(np.float16),
        meta_l[0].get_heatmap(target_size=(_network_w // _scale, _network_h // _scale)),
        meta_l[0].get_vectormap(target_size=(_network_w // _scale, _network_h // _scale)),
        meta_l[0].get_offsetmap(target_size=(_network_w // _scale, _network_h // _scale)),
        meta_l[0].get_heatmap_valid(target_size=(_network_w // _scale, _network_h // _scale)),
        meta_l[0].get_vectormap_valid(target_size=(_network_w // _scale, _network_h // _scale))
    ]
