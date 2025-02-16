import os
import json
import argparse
import pprint
parser = argparse.ArgumentParser()

parser.add_argument('--dir_name', type=str, default='bigcode/starcoder', help="")

args = parser.parse_args()
argsdict = vars(args)
print(pprint.pformat(argsdict))
dir_name = args.dir_name
tmp = {}
for json_file in os.listdir(dir_name):
    json_path = dir_name + f"/{json_file}"
    one_result = json.load(open(json_path))
    tmp[one_result[0]['base']] = one_result[0]


result = {'ifd_score':[], 'ifdx_score':[], 'inst_loss':[]}
order = sorted([int(x) for x in list(tmp.keys())])
# print(order)
for k in order:
    result['ifd_score'] += tmp[k]['ifd_score']
    result['ifdx_score'] += tmp[k]['ifdx_score']
    result['inst_loss'] += tmp[k]['inst_loss']
import math
for k in result:
    for i in range(len(result[k])):
        if math.isnan(result[k][i]):
            result[k][i] = 0
print(sum(result['ifd_score'])/len(result['ifd_score']))
print(sum(result['ifdx_score'])/len(result['ifdx_score']))
print(sum(result['inst_loss'])/len(result['inst_loss']))
json.dump(result, open(dir_name+'.json', 'w'))
