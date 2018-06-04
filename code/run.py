import argparse
import logging
import time
import ast

import common
import cv2
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--tag', type=str, default='blouse', help='blouse / outwear / trousers / skirt / dress')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    parser.add_argument('--test', type=str, default='not_submit', help='whether generate submit csv file')
    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model, args.tag), target_size=(w, h), clothe_class=args.tag)


    if os.path.isfile(args.image):
        if '.csv' in args.image and args.test == 'submit':
            print "----------------BEGIN SUBMIT TEST----------------"
            submit_file = '../submit/submit_%s.csv' % args.tag
            base_path = args.image.rsplit('/',1)[0]
            image_paths = []
            categories = []
            with open(args.image,'r') as f:
                lines = f.readlines()[1:]
                for line in lines:
                    elems = line.strip().split(',')
                    if elems[1] == args.tag:
                        image_paths.append(elems[0])
                        categories.append(elems[1])
            assert(len(image_paths) == len(categories))
            print("Get {} samples to test.".format(len(image_paths)))
            
            sub_image_paths = []
            sub_categories = []
            sub_parts = []
            if os.path.exists(args.image):
                for line in open(args.image,'r').readlines()[1:]:
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
                    parts = e.inference(image, args.tag, scales=scales, saveid=test_num)
                    assert(len(parts) == len(joint_idx_dict[args.tag]))
                    
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
            exit(0)
        if '.csv' in args.image:
            print "-------------BEGIN VAL---------------"
            base_path = '../data/val'
            total_score = 0
            image_paths = []
            categories = []
            partsets = []
            with open(args.image,'r') as f:
                lines = f.readlines()[1:]
                for line in lines:
                    elems = line.strip().split(',')
                    image_paths.append(elems[0])
                    categories.append(elems[1])
                    partset = []
                    for elem in elems[2:]:
                        x, y, visible = elem.split('_')
                        partset.append([int(x),int(y), int(visible)])
                    partsets.append(partset)
            assert(len(image_paths) == len(partsets))
            assert(len(image_paths) == len(categories))
            print("Get {} samples with {} parts.".format(len(image_paths), len(partsets[0])))
            
            test_num = 0
            total_num = 0
            parts_score = dict()
            parts_num = dict()
            # wf = open('val_id_image_name.txt', 'w')
            for image_path, category, partset in zip(image_paths, categories, partsets):
                if category == args.tag:
                    norm_dist = get_norm_dist(args.tag, partset)
                    if not norm_dist:
                        continue
                    image = common.read_imgfile(os.path.join(base_path, image_path), None, None)
                    
                    test_num += 1 
                    parts = e.inference(image, args.tag, scales=scales, saveid=test_num)
                    # wf.write('%s %d\n'%(image_path, test_num))
                    assert(len(parts) == len(joint_idx_dict[args.tag]))
                    
                    # for feature generation
                    # print test_num
                    # continue

                    cur_score = 0
                    cur_num = 0
                    to_show = []
                    full_to_show = []
                    for gt_id, gt in enumerate(partset):
                        if csv_seq[gt_id] not in joint_idx_dict[args.tag]:
                            full_to_show.append(0)
                            continue
                        assert( csv_seq[gt_id] in joint_idx_dict[args.tag] )
                        if gt[2] < 0:
                            to_show.append(0)
                            full_to_show.append(0)
                            continue 

                        pt = parts[joint_idx_dict[args.tag].index(csv_seq[gt_id])]
                        pt_score = get_dist(pt, gt) / norm_dist
                        
                        part_name = csv_seq[gt_id]
                        if part_name in parts_score.keys():
                            parts_score[part_name] += pt_score
                            parts_num[part_name] += 1
                        else:
                            parts_score[part_name] = pt_score
                            parts_num[part_name] = 1
                        #if test_num > 10:
                        #    print pt, gt
                        
 
                        if pt_score > 0.1:
                            to_show.append(1)
                            full_to_show.append(1)
                        else:
                            to_show.append(0)
                            full_to_show.append(0)

                        cur_score += pt_score
                        total_score += pt_score
                        cur_num += 1
                        total_num += 1
                    cur_score /= cur_num
                        
                    print("%05d with %g in 2134" % (test_num, cur_score))
                    # write image to show
                    # anno_image = image.copy()
                    # for part_id, part in enumerate(parts):
                    #     x, y, visible = part[:]
                    #     if visible >= 0 and to_show[part_id] == 1:
                    #         cv2.circle(image, (x,y), 2, (255,0,128), 2)
                    # for part_id, part in enumerate(partset):
                    #     x, y, visible = part[:]
                    #     if visible >= 0 and full_to_show[part_id] == 1:
                    #         cv2.circle(anno_image, (x,y), 2, (255,0,255), 2)
                    #         cv2.putText(anno_image, csv_seq[part_id], (x,y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,128))
                    # 
                    # if cur_score > 0.1:
                    #     cv2.imwrite('hard_images/%05d_%g.jpg'%(test_num,cur_score), image)
                    #     cv2.imwrite('anno_images/%05d.jpg'%test_num, anno_image)
                    # else:
                    #     cv2.imwrite('show_images/%05d_%g.jpg'%(test_num,cur_score), image)
            # wf.close()
            print("Test on {} samples : {}".format(test_num, total_score / total_num))
            for part_name in parts_score.keys():
                print("{}:{} @ {} got samples".format(part_name, parts_score[part_name]/parts_num[part_name], parts_num[part_name]))
        else:
            # estimate human poses from a single image !
            image = common.read_imgfile(args.image, None, None)
            # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            t = time.time()
            parts = e.inference(image, scales=scales)
            elapsed = time.time() - t

            logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

            for part in parts:
                x, y, visible = part[:]
                cv2.circle(image, (x,y), 2, (255,0,128), 2)
            cv2.imwrite('./show.jpg', image)
    if os.path.isdir(args.image):
        for filename in os.path.listdir(args.image):
            if '.jpg' in filename or '.png' in filename:
                full_path = os.path.join(args.image, filename)
                image = common.read_imgfile(args.image, None, None)
                parts = e.inference(image, scales=scales)        
                
    # image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    # # cv2.imshow('tf-pose-estimation result', image)
    # # cv2.waitKey()

    # import matplotlib.pyplot as plt

    # fig = plt.figure()
    # a = fig.add_subplot(2, 2, 1)
    # a.set_title('Result')
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

    # # show network output
    # a = fig.add_subplot(2, 2, 2)
    # plt.imshow(bgimg, alpha=0.5)
    # tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
    # plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    # plt.colorbar()

    # tmp2 = e.pafMat.transpose((2, 0, 1))
    # tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    # tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    # a = fig.add_subplot(2, 2, 3)
    # a.set_title('Vectormap-x')
    # # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    # plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    # plt.colorbar()

    # a = fig.add_subplot(2, 2, 4)
    # a.set_title('Vectormap-y')
    # # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    # plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    # plt.colorbar()
    # plt.show()

    # import sys
    # sys.exit(0)

    # logger.info('3d lifting initialization.')
    # poseLifting = Prob3dPose('./src/lifting/models/prob_model_params.mat')

    # image_h, image_w = image.shape[:2]
    # standard_w = 640
    # standard_h = 480

    # pose_2d_mpiis = []
    # visibilities = []
    # for human in humans:
    #     pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
    #     pose_2d_mpiis.append([(int(x * standard_w + 0.5), int(y * standard_h + 0.5)) for x, y in pose_2d_mpii])
    #     visibilities.append(visibility)

    # pose_2d_mpiis = np.array(pose_2d_mpiis)
    # visibilities = np.array(visibilities)
    # transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpiis, visibilities)
    # pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)

    # for i, single_3d in enumerate(pose_3d):
    #     plot_pose(single_3d)
    # plt.show()

    # pass
