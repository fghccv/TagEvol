import matplotlib.pyplot as plt
import numpy as np
import json

# 示例数据列表
data = json.load(open("ctf_oss_all_kcenter_72957_dist.json", "r"))

# 设定统计区间的数量
num_bins = 100

# 生成直方图数据
counts, bins = np.histogram(data, bins=num_bins)

# 绘制柱状图
plt.bar(bins[:-1], counts, width=np.diff(bins), edgecolor="black", align="edge")

# 设置图表标题和标签
plt.title('Data Distribution')
plt.xlabel('Value Range')
plt.ylabel('Frequency')

# 显示图表
plt.savefig("ctf_oss_all_kcenter_72957_dist.png")
