#数据集下载
from modelscope.msdatasets import MsDataset
import json
ds =  MsDataset.load('modelscope/gsm8k', subset_name='main', split='test')
#您可按需配置 subset_name、split，参照“快速使用”示例代码
print(ds[0])
datas = [{'instruction':ds[i]['question'], 'output':ds[i]['answer']} for i in range(len(ds))]
json.dump(datas, open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math/math_evaluate/datas/gsm8k.json", 'w'))

