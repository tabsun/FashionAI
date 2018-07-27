import argparse
import time

import common
import cv2
import os
import math
import numpy as np

#from lifting.prob_model import Prob3dPose
#from lifting.draw import plot_pose

csv_seq = list(['nkl','nkr','cf','shl','shr','apl','apr','wll','wlr','cli','clo','cri','cro','thl','thr','wbl','wbr','hll','hlr','cr','bli','blo','bri','bro'])
joint_idx_dict = dict()
joint_idx_dict['blouse'] = list(['nkl','nkr','shl','shr','cf','apl','apr','thl','thr','cli','clo','cri','cro'])
joint_idx_dict['dress'] = list(['nkl','nkr','shl','shr','cf','apl','apr','wll','wlr','cli','clo','cri','cro','hll','hlr'])
joint_idx_dict['skirt'] = list(['wbl','wbr','hll','hlr'])
joint_idx_dict['trousers'] = list(['wbl','wbr','cr','bli','blo','bri','bro'])
joint_idx_dict['outwear'] = list(['nkl','nkr','shl','shr','apl','apr','wll','wlr','cli','clo','cri','cro','thl','thr'])
link_idx_dict = dict()
link_idx_dict['blouse'] = list([['nkr','nkl'],
                                ['nkl','shl'],
                                ['nkr','shr'],
                                ['nkl','cf'],
                                ['nkr','cf'],
                                ['shl','clo'],
                                ['shr','cro'],
                                ['clo','cli'],
                                ['cro','cri'],
                                ['cli','apl'],
                                ['cri','apr'],
                                ['apl','thl'],
                                ['apr','thr'],
                                ['thl','thr']])

link_idx_dict['outwear'] = list([['nkr','nkl'],
                                ['nkl','shl'],
                                ['nkr','shr'],
                                ['shl','clo'],
                                ['shr','cro'],
                                ['clo','cli'],
                                ['cro','cri'],
                                ['cli','apl'],
                                ['cri','apr'],
                                ['apl','wll'],
                                ['apr','wlr'],
                                ['wll','thl'],
                                ['wlr','thr'],
                                ['wll','wlr'],
                                ['thl','thr']])
link_idx_dict['dress'] = list([ ['nkr','nkl'],
                                ['nkl','shl'],
                                ['nkr','shr'],
                                ['nkl','cf' ],
                                ['nkr','cf' ],
                                ['shl','clo'],
                                ['shr','cro'],
                                ['clo','cli'],
                                ['cro','cri'],
                                ['cli','apl'],
                                ['cri','apr'],
                                ['apl','wll'],
                                ['apr','wlr'],
                                ['wll','hll'],
                                ['wlr','hlr'],
                                ['wll','wlr'],
                                ['hll','hlr']])
link_idx_dict['trousers'] = list([['wbl','wbr'],
                                ['wbl','blo'],
                                ['wbr','bro'],
                                ['blo','bli'],
                                ['bro','bri'],
                                ['bli','cr' ],
                                ['bri','cr' ] ])
link_idx_dict['skirt'] = list([ ['wbl','wbr'],
                                ['hll','wbl'],
                                ['wbr','hlr'],
                                ['hll','hlr'] ])

norm_index = {'blouse':['apl','apr'], 'skirt':['hll','hlr'], 'dress':['apl','apr'], 'trousers':['blo','bro'], 'outwear':['apl','apr']}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--submit', type=str, default='./submit.csv')
    parser.add_argument('--imagepath', type=str, default='./images/p1.jpg')
    parser.add_argument('--tag', type=str, default='blouse', help='blouse / outwear / trousers / skirt / dress')
    parser.add_argument('--outputpath', type=str, default='./')

    args = parser.parse_args()

    test_num = 0
    if '.csv' in args.submit and os.path.exists(args.submit):
        with open(args.submit,'r') as f:
            for line in open(args.submit,'r').readlines()[1:]:
                elems = line.strip().split(',')
                image_path = os.path.join(args.imagepath,elems[0])
                category = elems[1]
                if category == args.tag or args.tag == 'all':
                    assert(len(elems) == 2+len(csv_seq))
                    parts = []
                    for elem in elems[2:]:
                       parts.append([int(x) for x in elem.split('_')])
                    # plot image
                    links = link_idx_dict[category]
                    image = cv2.imread(image_path)
                    for part in parts:
                        x,y,visible = part[:]
                        if(visible == 1):
                            cv2.circle(image, (x,y), 2, (255,0,128), 2)
                        elif(visible == 0):
                            cv2.circle(image, (x,y), 3, (0,128,255), 2)
                    for link in links:
                        pt1, pt2 = parts[csv_seq.index(link[0])], parts[csv_seq.index(link[1])]
                        cv2.line(image, (pt1[0],pt1[1]), (pt2[0],pt2[1]),(255,225,0),2 )
                    # save image
                    cv2.imwrite(os.path.join(args.outputpath, "%s_%05d.jpg"%(category, test_num)), image)
                    test_num += 1
                    print test_num
