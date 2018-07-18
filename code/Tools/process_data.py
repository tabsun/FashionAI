import argparse
import logging
import time
import ast

import common
import cv2
import nudged
import os
import math
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
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

norm_index = {'blouse':['apl','apr'], 'skirt':['wbl','wbr'], 'dress':['apl','apr'], 'trousers':['wbl','wbr'], 'outwear':['apl','apr']}
def get_dist(pt, gt):
    pt_x, pt_y, pt_v = pt[:]
    gt_x, gt_y, gt_v = gt[:]
    if pt_v != 1:
        return math.sqrt(gt_x**2 + gt_y**2)
    if pt_v == 1:
        return math.sqrt((gt_x-pt_x)**2 + (gt_y-pt_y)**2)
    assert("Other sitiation???")

def get_norm_dist(clothe_class, parts):
    part_name1, part_name2 = norm_index[clothe_class][:]
    id1, id2 = csv_seq.index(part_name1), csv_seq.index(part_name2)
    if(parts[id1][2] >= 0 and parts[id2][2] >= 0):
        return get_dist(parts[id1], parts[id2])
    else:
        return None

def reseq(parts, clothe_class):
    joints = joint_idx_dict[clothe_class]
    nparts = []
    for part_name in csv_seq:
        if part_name in joints:
            part = parts[joints.index(part_name)]
        else:
            part = [-1,-1,-1]
        nparts.append(part)
    return nparts

def get_bounding_box(pt_array, scale_x, scale_y):
    # dress strategy
    # left, right = int(np.min(pt_array[:-2,0])), int(np.max(pt_array[:-2,0]))
    # top, bottom = int(np.min(pt_array[:,1])), int(np.max(pt_array[:,1]))
    left, right = int(np.min(pt_array[:,0])), int(np.max(pt_array[:,0]))
    top,  bottom= int(np.min(pt_array[:,1])), int(np.max(pt_array[:,1]))
    width = right - left + 1
    height = bottom - top + 1
    left   -= width  * (scale_x-1.0)/2.0
    right  += width  * (scale_x-1.0)/2.0
    top    -= height * (scale_y-1.0)/2.0
    bottom += height * (scale_y-1.0)/2.0
    width = right - left + 1
    height = bottom - top + 1
    left = max(0, left)
    top = max(0, top)
    return int(left), int(right), int(top), int(bottom)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--tag', type=str, default='blouse', help='blouse / outwear / trousers / skirt / dress')
    args = parser.parse_args()

    outf = open('../data/align/%s.csv'%args.tag, 'w')
    if os.path.isfile(args.image):
        if '.csv' in args.image:
            print "-------------BEGIN PROCESS---------------"
            base_path = '../data/train'
            joint_shape = joint_idx_dict[args.tag]
            image_paths = []
            categories = []
            partsets = []
            with open(args.image,'r') as f:
                lines = f.readlines()[1:]
                for line in lines:
                    elems = line.strip().split(',')
                    if elems[1] != args.tag:
                        continue
                    
                    partset = []
                    full_partset = []
                    for part_name, elem in zip(csv_seq,elems[2:]):
                        x, y, visible = elem.split('_')
                        full_partset.append([int(x),int(y), int(visible)])
                    for part_name in joint_shape:
                        cur_pt = full_partset[csv_seq.index(part_name)]
                        if cur_pt[0] < 0 or cur_pt[1] < 0 or cur_pt[2] < 0:
                            continue 
                        partset.append(cur_pt)
                    if len(partset) != len(joint_shape):
                        continue
                    partsets.append(partset)
                    image_paths.append(os.path.join(base_path, elems[0]))
                    categories.append(elems[1])
            assert(len(image_paths) == len(partsets))
            assert(len(image_paths) == len(categories))
            assert(len(partsets[0]) == len(joint_shape))
            print("Get {} samples with {} parts.".format(len(image_paths), len(partsets[0])))
            
            # anchors
            anchor_parts = anchor_parts_dict[args.tag]
            anchor_shape = anchor_shapes_dict[args.tag]
            
            # transform
            output_base_path = '../data/align/%s' % args.tag
            test_num = 0
            #for test_num in range(len(image_paths)):
            #    image_path, category, partset = image_paths[test_num], categories[test_num], partsets[test_num]
            for image_path, category, partset in zip(image_paths, categories, partsets):
                if category == args.tag:
                    image = cv2.imread(image_path)
                    h, w = image.shape[:2]
                    std_size = int(math.sqrt(w**2 + h**2) + 0.5)
                    full_image = np.zeros((std_size, std_size, 3), dtype=np.float32)
                    offset_x, offset_y = (std_size - w) / 2, (std_size - h) / 2
                    full_image[offset_y:(offset_y+h),offset_x:(offset_x+w),:] = image
                    # modify the ground_truth points
                    for i in range(len(partset)):
                        partset[i][0] += offset_x
                        partset[i][1] += offset_y

                    # filter the anchor ground_truth points
                    anchor_partset = []
                    for part_name, pt in zip(joint_shape, partset):
                        if part_name in anchor_parts:
                            anchor_partset.append(pt[:2])
                    anchor_partset = np.array(anchor_partset).reshape((-1,2))
                    assert(anchor_partset.shape == anchor_shape.shape)
                    
                    
                    # align the image and points
                    tform = nudged.estimate(anchor_partset, anchor_shape)
                    tform_array = np.asarray(tform.get_matrix())
                    image_align = cv2.warpPerspective(full_image, tform_array, (1000,1000))

                    new_partset = []
                    new_anchor_partset = []
                    for part_name, pt in zip(joint_shape, partset):
                        x, y = tform.transform([pt[0], pt[1]])
                        new_pt = [x, y, pt[2]]
                        new_partset.append(new_pt)
                        if part_name in anchor_parts:
                            new_anchor_partset.append(new_pt)
                    
                    new_anchor_shape = np.array(new_anchor_partset)[:,:2]
                    
                    # crop and resize the roi image
                    # outwear / dress   2.2 1.4 1.5
                    # blouse            1.6 1.4 1.2
                    # skirt / trousers  1.2 1.2 1.8
                    if args.tag == 'outwear' or 'dress':
                        extend_scale_w = 2.2
                        extend_scale_h = 1.4
                        std_ratio      = 1.5
                    if args.tag == 'skirt' or 'trousers':
                        extend_scale_w = 1.6
                        extend_scale_h = 1.4
                        std_ratio      = 1.2
                    if args.tag == 'blouse':
                        extend_scale_w = 1.8
                        extend_scale_h = 1.4
                        std_ratio      = 1.4
                    #left, right, top, bottom = get_bounding_box(new_anchor_shape, extend_scale_w, extend_scale_h)
                    left, right, top, bottom = get_bounding_box(np.array(new_partset)[:,:2], 1.2, 1.2)
                    scale = (bottom-top)*1.0/(right-left)
                    if scale > std_ratio:
                        # extend width
                        input_h = bottom - top + 1
                        input_w = int(input_h / std_ratio)
                    else:
                        # extend height
                        input_w = right - left + 1
                        input_h = int(input_w * std_ratio)
                    # specifical strategy for only dress category
                    train_std_image = image_align[top:bottom, left:right, :]
                    full_std_image = np.zeros((input_h, input_w, 3), dtype=np.float32)
                    std_w, std_h = right-left, bottom-top
                    off_x = (input_w - std_w) / 2
                    off_y = (input_h - std_h) / 2
                    full_std_image[off_y:(off_y+std_h),off_x:(off_x+std_w),:] = train_std_image
                    # crop the points
                    train_std_partset = []
                    for pt in new_partset:
                        train_std_partset.append( [pt[0] - left + off_x, pt[1] - top + off_y, pt[2]] )
                    
                    h, w = full_std_image.shape[:2]
                    assert(abs(h * 1.0 / w - std_ratio) < 0.05)
                    scale = 368.0 / w
                    for i in range(len(new_partset)):
                        train_std_partset[i][0] = int(train_std_partset[i][0]*scale)
                        train_std_partset[i][1] = int(train_std_partset[i][1]*scale)

                    save_image = cv2.resize( full_std_image, (368,int(368*std_ratio)) )
                    # for pt in train_std_partset:
                    #     cv2.circle(full_std_image, (int(pt[0]),int(pt[1])), 5, (255,0,128), -1)
                    
                    cv2.imwrite(os.path.join(output_base_path, "%05d.jpg"%test_num), save_image)
                    cur_line = "%s/%05d.jpg,%s" % (args.tag, test_num, args.tag)
                    
                    for part_name in csv_seq:
                        if part_name in joint_shape:
                            pt = train_std_partset[joint_shape.index(part_name)]
                            cur_line += ",%d_%d_%d" % (pt[0], pt[1], pt[2])
                        else:
                            cur_line += ",-1_-1_-1"
                    cur_line += '\n'
                    outf.write(cur_line)
                    test_num += 1
                    
                    print "%d in %d" % (test_num, len(image_paths))
    outf.close()
