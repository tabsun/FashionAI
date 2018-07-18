import argparse
import logging
import time
import ast

import common
import cv2
import dlib
import os
import math
import numpy as np
from random import sample

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

#from lifting.prob_model import Prob3dPose
#from lifting.draw import plot_pose

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
joint_idx_dict['outwear2nd'] = list(['apl','wll','thl','thr','wlr','apr'])
#link_idx_dict = dict()
#link_idx_dict['blouse'] = list([['nkr','nkl'],
#                                ['nkl','shl'],
#                                ['nkr','shr'],
#                                ['nkl','cf'],
#                                ['nkr','cf'],
#                                ['shl','clo'],
#                                ['shr','cro'],
#                                ['clo','cli'],
#                                ['cro','cri'],
#                                ['cli','apl'],
#                                ['cri','apr'],
#                                ['apl','thl'],
#                                ['apr','thr'],
#                                ['thl','thr']])
#
#link_idx_dict['outwear'] = list([['nkr','nkl'],
#                                ['nkl','shl'],
#                                ['nkr','shr'],
#                                ['shl','clo'],
#                                ['shr','cro'],
#                                ['clo','cli'],
#                                ['cro','cri'],
#                                ['cli','apl'],
#                                ['cri','apr'],
#                                ['apl','wll'],
#                                ['apr','wlr'],
#                                ['wll','thl'],
#                                ['wlr','thr'],
#                                ['wll','wlr'],
#                                ['thl','thr']])
#link_idx_dict['outwear2nd'] = list([['apl','wll'],
#                                    ['wll','thl'],
#                                    ['thl','thr'],
#                                    ['thr','wlr'],
#                                    ['wlr','apr'],
#                                    ['apr','apl'],
#                                    ['wll','wlr']])
#
#link_idx_dict['dress'] = list([ ['nkr','nkl'],
#                                ['nkl','shl'],
#                                ['nkr','shr'],
#                                ['nkl','cf' ],
#                                ['nkr','cf' ],
#                                ['shl','clo'],
#                                ['shr','cro'],
#                                ['clo','cli'],
#                                ['cro','cri'],
#                                ['cli','apl'],
#                                ['cri','apr'],
#                                ['apl','wll'],
#                                ['apr','wlr'],
#                                ['wll','hll'],
#                                ['wlr','hlr'],
#                                ['wll','wlr'],
#                                ['hll','hlr']])
#link_idx_dict['trousers'] = list([['wbl','wbr'],
#                                ['wbl','blo'],
#                                ['wbr','bro'],
#                                ['blo','bli'],
#                                ['bro','bri'],
#                                ['bli','cr' ],
#                                ['bro','cr' ] ])
#link_idx_dict['skirt'] = list([ ['wbl','wbr'],
#                                ['hll','wbl'],
#                                ['wbr','hlr'],
#                                ['hll','hlr'] ])

second_model_parts = {'blouse':  ['shl','shr','thl','thr','apl'], 
                      'dress':   ['apr','cli','hll','hlr'], 
                      'trousers':['bli','cr'], 
                      'skirt':   ['wbr'], 
                      'outwear': ['wll','wlr']}
norm_index = {'blouse':['apl','apr'], 'skirt':['wbl','wbr'], 'dress':['apl','apr'], 'trousers':['wbl','wbr'], 'outwear':['apl','apr'],'outwear2nd':['apl','apr']}
def get_dist(pt, gt):
    pt_x, pt_y, pt_v = pt[:]
    gt_x, gt_y, gt_v = gt[:]
    return math.sqrt((gt_x-pt_x)**2 + (gt_y-pt_y)**2)

def get_norm_dist(clothe_class, parts):
    part_name1, part_name2 = norm_index[clothe_class][:]
    id1, id2 = joint_idx_dict[clothe_class].index(part_name1), joint_idx_dict[clothe_class].index(part_name2)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--csv', type=str, default='./images/p1.jpg')
    parser.add_argument('--imagepath', type=str, default='../data/train/Images')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--modelpath', type=str, default='./models/trained/frozen_graph.pb', help='the path of frozen_*.pb')
    parser.add_argument('--tag', type=str, default='blouse', help='blouse / outwear / trousers / skirt / dress')
    parser.add_argument('--inputsize', type=str, default='512', help='368 or 512')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    parser.add_argument('--test', type=str, default='validate', help='submit or validate')
    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)

    w, h = model_wh(args.resolution)
    if not os.path.exists(args.modelpath):
        print "{} do not exists!".format(args.modelpath)

    e = TfPoseEstimator(args.modelpath, target_size=(w,h), clothe_class=args.tag)
    #colors = [(0,0,0), (0,0,128), (0,0,255), (0,128,0), (0,128,128), 
    #          (0,128,255), (0,255,0), (0,255,128), (0,255,255), (128,0,0), 
    #          (128,0,128), (128,0,255), (128,128,0), (128,128,128), (128,128,255),
    #          (128,255,0), (128,255,128),(128,255,255),(255,0,0),(255,0,128),
    #          (255,0,255), (255,128,0), (255,128,128),(255,128,255)]

    if os.path.isfile(args.csv):
        if '.csv' in args.csv and args.test == 'submit':
            print "----------------BEGIN SUBMIT TEST----------------"
            submit_file = '../submit/submit_%s.csv' % args.tag
            base_path = args.imagepath
            image_paths = []
            categories = []
            with open(args.csv,'r') as f:
                lines = f.readlines()[1:]
                for line in lines:
                    elems = line.strip().split(',')
                    if elems[1] == args.tag:
                        image_paths.append(elems[0])
                        categories.append(elems[1])
            assert(len(image_paths) == len(categories))
            print("Get {} samples to test.".format(len(image_paths)))
            
            joint = joint_idx_dict[args.tag]
            #link = link_idx_dict[args.tag]
            # read the detected-done samples to wait for new output
            sub_image_paths = []
            sub_categories = []
            sub_parts = []
            if os.path.exists(submit_file):
                for line in open(submit_file,'r').readlines()[1:]:
                    elems = line.strip().split(',')
                    sub_image_paths.append(elems[0])
                    sub_categories.append(elems[1])
                    if len(elems) > 3:
                        assert(len(elems) == 2+len(csv_seq))
                        one_sub_part = []
                        for elem in elems[2:]:
                           one_sub_part.append([int(x) for x in elem.split('_')])
                        sub_parts.append(one_sub_part)
                    else:
                        sub_parts.append([])

            test_num = 0
            for image_path, category in zip(image_paths, categories):
                if category == args.tag:
                    image = common.read_imgfile(os.path.join(base_path, image_path), None, None)
                    
                    test_num += 1 
                    parts = e.inference(image, args.tag, int(args.inputsize), scales=scales, saveid=test_num)

                    assert(len(parts) == len(joint_idx_dict[args.tag]))

                    #anno_image = image.copy()
                    #for part_id, pd_pt in enumerate(parts):
                    #    color = colors[csv_seq.index(joint[part_id])]
                    #    cv2.circle(anno_image, (pd_pt[0],pd_pt[1]), 5, color, -1)
                    #for part_pair in link:
                    #    cur_pt, next_pt = parts[joint.index(part_pair[0])], parts[joint.index(part_pair[1])]
                    #    cv2.line(anno_image, (cur_pt[0],cur_pt[1]), (next_pt[0],next_pt[1]), (128,128,0),2)

                    #cv2.imwrite(os.path.join('twin_images/%s'%args.tag, '%05d.jpg' % test_num), anno_image)                    
                    # reseq to csv sequence and pad [-1,-1,-1]
                    csv_parts = reseq(parts, args.tag)
                    global_id = sub_image_paths.index(image_path)
                    assert(sub_categories[global_id] == category)
                    sub_parts[global_id] = csv_parts
                    print("Testing on {} sample / {}".format(test_num, len(image_paths)))
            # update submit file
            if os.path.exists(submit_file):
                os.remove(submit_file)
            with open(submit_file,'w') as f:
                f.write('image_id,image_category,neckline_left,neckline_right,center_front,shoulder_left,shoulder_right,armpit_left,armpit_right,waistline_left,waistline_right,cuff_left_in,cuff_left_out,cuff_right_in,cuff_right_out,top_hem_left,top_hem_right,waistband_left,waistband_right,hemline_left,hemline_right,crotch,bottom_left_in,bottom_left_out,bottom_right_in,bottom_right_out\n')
                for image_path, category, parts in zip(sub_image_paths, sub_categories, sub_parts):
                    parts_str = ""
                    for part in parts:
                        parts_str += "{}_{}_{},".format(part[0],part[1],part[2])
                    parts_str = parts_str[:-1]
                    f.write('%s,%s,%s\n' %(image_path, category, parts_str))

        # Validation Test
        if '.csv' in args.csv and args.test == 'validate':
            print "-------------BEGIN VAL---------------"
            base_path = args.imagepath
            total_score = 0
            image_paths = []
            categories = []
            gts = []
            with open(args.csv,'r') as f:
                lines = f.readlines()[1:]
                for line in lines:
                    elems = line.strip().split(',')
                    if elems[1] != args.tag:
                        continue
                    image_paths.append(elems[0])
                    categories.append(elems[1])
                    partset = []
                    for part_name in joint_idx_dict[args.tag]:
                        elem_id = csv_seq.index(part_name) + 2
                        x, y, visible = elems[elem_id].split('_')
                        partset.append([int(x),int(y), int(visible)])
                    gts.append(partset)
            assert(len(image_paths) == len(gts))
            assert(len(image_paths) == len(categories))
            print("Get {} samples with {} parts.".format(len(image_paths), len(gts[0])))
 
            test_num = 0
            total_num = 0
            skip_num = 0
            parts_score = dict()
            parts_num = dict()
 
            status_scores = [0,0,0]
            status_cnt    = [0,0,0]
            for image_path, category, gt in zip(image_paths, categories, gts):
                if category == args.tag:
                    norm_dist = get_norm_dist(args.tag, gt)
                    if not norm_dist:
                        continue
                    image = common.read_imgfile(os.path.join(base_path, image_path), None, None)
                    
                    pd = e.inference(image, args.tag, int(args.inputsize), scales=scales, saveid=test_num)

                    assert(len(pd) == len(joint_idx_dict[args.tag]))
 
                    cur_score = 0
                    cur_num = 0
                    anno_image = image.copy()
                    for pd_pt, gt_pt, part_name in zip(pd, gt, joint_idx_dict[args.tag]):
                        # not need points
                        if gt_pt[2] <= 0:
                            continue 

                        pt_score = get_dist(pd_pt, gt_pt) / norm_dist
                        if pd_pt[2] == -1:
                            status_scores[0] += pt_score
                            status_cnt[0] += 1
                        elif pd_pt[2] == 0:
                            status_scores[1] += pt_score
                            status_cnt[1] += 1
                        elif pd_pt[2] == 1:
                            status_scores[2] += pt_score
                            status_cnt[2] += 1
                        else:
                            print 'status != -1 0 or 1, How could it be?'
                            exit(0)
                        if part_name in parts_score.keys():
                            parts_score[part_name] += pt_score
                            parts_num[part_name] += 1
                        else:
                            parts_score[part_name] = pt_score
                            parts_num[part_name] = 1
                        
                        cur_score += pt_score
                        total_score += pt_score
                        cur_num += 1
                        total_num += 1
                    cur_score /= (cur_num+0.00000001)

                    image_name = image_path.split('/')[-1]    

                    print("%05d with %g in %d" % (test_num, cur_score, len(gts)))
                    test_num += 1

            print("Test on {} samples : {}".format(test_num, total_score / total_num))
            print "invalid but max choose: %g with %d contribute %g" % (status_scores[0]  / (status_cnt[0]+0.0000001), status_cnt[0], status_scores[0])
            print "fullfill by mean shape: %g with %d contribute %g" % (status_scores[1]  / (status_cnt[1]+0.0000001), status_cnt[1], status_scores[1])
            print "valid detected by model: %g with %d contribute %g" % (status_scores[2] / (status_cnt[2]+0.0000001), status_cnt[2], status_scores[2])
            for part_name in parts_score.keys():
                print("{}:{} @ {} got samples".format(part_name, parts_score[part_name]/parts_num[part_name], parts_num[part_name]))

