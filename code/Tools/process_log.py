import os
import ast
import argparse
sample_nums = dict()
sample_nums['blouse'] = {'shr' : 0.08339845,
                         'clo' : 0.08822902,
                         'cli' : 0.08188889,
                         'apr' : 0.04320878,
                         'nkl' : 0.07704056,
                         'cf' : 0.089916175,
                         'thr' : 0.08062797,
                         'nkr' : 0.07691624,
                         'thl' : 0.08105420,
                         'apl' : 0.04361724,
                         'cri' : 0.08227960,
                         'shl' : 0.08386019,
                         'cro' : 0.08796263}
sample_nums['dress'] = {'shr' : 0.07262504,
                        'clo' : 0.06451829,
                        'cli' : 0.05565626,
                        'apr' : 0.05283652,
                        'wlr' : 0.05365894,
                        'nkl' : 0.07262504,
                        'cf' : 0.084944612,
                        'hlr' : 0.08472641,
                        'nkr' : 0.07309499,
                        'hll' : 0.08472641,
                        'wll' : 0.05456529,
                        'apl' : 0.05391070,
                        'cri' : 0.05548841,
                        'shl' : 0.07232292,
                        'cro' : 0.06430010}
sample_nums['outwear'] = {'shr' : 0.085927,
                          'clo' : 0.094968,
                          'cli' : 0.085121,
                          'apr' : 0.040310,
                          'wlr' : 0.016686,
                          'nkl' : 0.084414,
                          'thr' : 0.092865,
                          'nkr' : 0.083392,
                          'thl' : 0.092865,
                          'wll' : 0.017177,
                          'apl' : 0.040723,
                          'cri' : 0.084316,
                          'shl' : 0.087047,
                          'cro' : 0.094182}
sample_nums['skirt'] = {'wbl' : 0.22125343,
                        'hll' : 0.27858803,
                        'wbr' : 0.22141196,
                        'hlr' : 0.27874656}
sample_nums['trousers'] = {'wbr' : 0.11432,
                           'bro' : 0.15565,
                           'blo' : 0.15627,
                           'bli' : 0.15256,
                           'bri' : 0.15305,
                           'cr' : 0.151942,
                           'wbl' : 0.11618}
#sample_nums['trousers'] = {'wbr' : 0.1141309,
#                           'bro' : 0.1560130,
#                           'blo' : 0.1560439,
#                           'bli' : 0.1532889,
#                           'bri' : 0.1532270,
#                           'cr' : 0.15078161,
#                           'wbl' : 0.1165144}


def get_result_dict(lines, start, end, filename_tag):
    seq = lines[start+1].strip()
    segment = lines[start:end+1]
    for index, line in enumerate(segment):
        if 'Test on' in line:
            start = index
            break
    segment = segment[start:]
    result = dict()
    result['seq'] = '%s_%s' % (filename_tag, seq)
    result['partnames'] = []
    result['num'] = 0
    for line in segment:
        if len(line.split(':')[0]) == 3 or len(line.split(':')[0]) == 2:
            part_name = line.split(':')[0]
            score = float(line.split(':')[1].split('@')[0])
            cur_num = float(line.split('@')[1].split('got samples')[0])
            #score *= cur_num
            #score *= sample_num[part_name]
            result['partnames'].append(part_name)
            result[part_name] = score
            #result['%s_num'%part_name] = sample_num
            result['num'] += cur_num

    if len(result['partnames']) == 0:
        return None
    else:
        return result

def get_score(ri, rj):
    assert(set(ri['partnames']) == set(rj['partnames']))
    #assert(ri['num'] == rj['num'])
    part_names = ri['partnames']
    score = 0
    i_part_names = []
    j_part_names = []
    for part_name in part_names:
        if ri[part_name] < rj[part_name]:
            i_part_names.append(part_name)
            score += ri[part_name]*sample_num[part_name]
        else:
            j_part_names.append(part_name)
            score += rj[part_name]*sample_num[part_name]
    return score, i_part_names, j_part_names
    
def extract_results(filename, filename_tag):
    with open(filename,'r') as f:
        lines = f.readlines()

    flags = []
    for line_id, line in enumerate(lines):
        if '=================' in line:
            flags.append(line_id)
    flags.append(len(lines)-1)
    
    results = []
    for index, start in enumerate(flags):
        if index == len(flags)-1:
            break
    
        end = flags[index+1]
        one_result = get_result_dict(lines, start, end, filename_tag)
        if one_result:
            results.append(one_result)
    return results

def get_result_by_name(results, name):
    for result in results:
        if result['seq'] == name:
            return result
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model choose')
    parser.add_argument('--tag', type=str, default='blouse')
    parser.add_argument('--flags', type=str, default='[None]')
    args = parser.parse_args()
    flags = ast.literal_eval(args.flags)

    tag = args.tag
    if not isinstance(flags, list):
        print "--flags should be a list."
        exit(0)

    sample_num = sample_nums[tag]
    results = []
    
    for flag in flags:
        filename = "%s_%s.txt"%(tag, flag)
        if not os.path.exists(filename):
            print "%s do not exist." % filename
            exit(0)

        filename_tag = filename.split('.')[0]
        results += extract_results(filename, filename_tag)
    
    total_num = len(results)
    print "Totally we have {} model choices".format(total_num)
    best_score = 100000000.0
    best_i_parts = None
    best_j_parts = None
    best_pair = None
    for i in range(total_num-1):
        ri = results[i]
        for j in range(i+1, total_num):
            rj = results[j]
            score, i_parts, j_parts = get_score(ri, rj)
            if score < best_score:
                best_i_parts = i_parts
                best_j_parts = j_parts
                best_pair = [ri['seq'], rj['seq']]
                best_score = score
    
    ri_name, rj_name = best_pair[:]
    ri, rj = get_result_by_name(results, ri_name), get_result_by_name(results, rj_name)
    print "You can use {} + {}".format(ri['seq'], rj['seq'])
    print "With following from {}:".format(ri['seq'])
    for part_name in best_i_parts:
        print "{}: mean {}".format(part_name, ri[part_name])
    print "With following from {}:".format(rj['seq'])
    for part_name in best_j_parts:
        print "{}: mean {}".format(part_name, rj[part_name])
    print "Get best score {}".format(best_score)
