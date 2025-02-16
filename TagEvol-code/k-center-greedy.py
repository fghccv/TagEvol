import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import json
import random
 
class KCenterGreedy:
    def __init__(self, k, centers):
        self.k = k
        self.centers = centers
        print(len(self.centers))   
    def fit(self, data):
        data = data.to('cuda')

        bar = tqdm.tqdm(total=k)
        if self.centers == []:
            init_index = random.choice(range(data.shape[0]))
            self.centers.append(init_index)
            bar.update(1)
        else:
            bar.update(len(self.centers))
        while len(self.centers) < self.k:
            # 计算数据集中每个点到其最近中心点的最小距离
            centers = data[self.centers].to('cuda')
            cos_sim=data@centers.T
            cos_sim /= torch.norm(data, p=2, dim=1,keepdim=True) @ torch.norm(centers, p=2, dim=1,keepdim=True).T
            dist_to_closest_center, _ = torch.max(cos_sim, dim=1, keepdim=False)
            # 选择距离现有中心点最远的点
            new_index = torch.argmin(dist_to_closest_center)
            self.centers.append(int(new_index.item()))
            bar.update(1)
 
    def get_center(self):
        return list(set(self.centers))
 
def plot_k_center_greedy(points, centers, path):
    plt.scatter(points[:, 0], points[:, 1], c='blue', label='Data Points')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Center Points')
    plt.title('K-Center-Greedy Result')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.show()
    plt.savefig(path)
 
 
if __name__ == '__main__':
    random.seed(42)
    points = torch.load("/home/zhoushiqi/workplace/icaed/embeddings/bge-en-icl/embeddings.pth")
    points = torch.tensor(points)
    all_indexs = {}
    print(all_indexs.keys())
    centers_index = []
    for k in [len(points)//10*(i+1) for i in range(0,9)]:
 
        kc_greedy = KCenterGreedy(k, centers_index)
        kc_greedy.fit(points)
    
        centers_index = kc_greedy.centers
        all_indexs[k] = kc_greedy.get_center()
        centers = points[centers_index]
        # print("Center points:")
        # print(kc_greedy.get_center(), len(kc_greedy.get_center()))
        # 绘制结果
        plot_k_center_greedy(points, centers, f"/home/zhoushiqi/workplace/icaed/icae/k-center-greedy/fig/fig_k{k}.png")
        json.dump(all_indexs, open("/home/zhoushiqi/workplace/icaed/icae/k-center-greedy/all_index.json", 'w'))

