import os 
import sys
import yaml
import torch 
import numpy as np 
from tqdm import tqdm 
from sklearn.cluster import KMeans
from easydict import EasyDict

config_path = sys.argv[1]
#config_path = "/data12/zzf/MIL/AAAI_github/CIMIL/config/config.yml"
with open(config_path, 'r') as f:
    config = EasyDict(yaml.safe_load(f))
train_pyg_dir = config.cluster.train_dir 
ks = config.cluster.ks
seed = config.cluster.seed

# def gen_idx(indexes,ks):
#     # [5,4,3]; [10,5,4] 0.02 
    
#     assert len(indexes)>0
#     c = g['cluster_level1_'+str(ks[0])]==indexes[0]
#     for li in range(1,len(indexes)):
#         c = c&(g[f'cluster_level{li+1}_{ks[li]}']==indexes[li])
#     return c

# def cluster_recursive(ks,level,indexes):
#     print(indexes,level)
#     if len(ks)==0:
#         return 
#     if indexes==-1:
#         c = torch.ones(len(g.x)).cpu().bool()
#     else:
#         c = gen_idx(indexes,ks[0:])
#     k_num_now = min(ks[0:][0],len(g.x[c]))
#     print(c,k_num_now)
#     kmean = KMeans(n_clusters=k_num_now, random_state=0)
#     cluster = kmean.fit(g.x[c])
#     g[f'cluster_level{level}_{k_num_now}'][c] = torch.LongTensor(cluster.labels_).cpu()
#     for ki in range(k_num_now):
#         if indexes==-1:
#             cluster_recursive(ks,level+1,[ki])
#         else:
#             cluster_recursive(ks,level+1,indexes.append(ki))
#     return 



# for pkl in tqdm(os.listdir(train_pyg_dir)):
#     g = torch.load(os.path.join(train_pyg_dir,pkl))
#     level = 1
#     for ki in ks:
#         g[f'cluster_level_{level}_{ki}']=-torch.ones(len(g.x)).cpu().long()
#         level+=1
#     del level
#     print(ks[:config.cluster.pre_max_level])
#     cluster_recursive(ks[:config.cluster.pre_max_level],1,-1)


import os 
import sys
import yaml
import torch 
import numpy as np 
from tqdm import tqdm 
from sklearn.cluster import KMeans
from easydict import EasyDict


config_path = sys.argv[1]
#config_path = "/data12/zzf/MIL/AAAI_github/CIMIL/config/config.yml"
with open(config_path, 'r') as f:
    config = EasyDict(yaml.safe_load(f))
train_pyg_dir = config.cluster.train_dir 
ks = config.cluster.ks
seed = config.cluster.seed

for i in [ks[0]]:
    c0 = KMeans(n_clusters=i, random_state=seed)
    c = KMeans(n_clusters=ks[1], random_state=seed)
    for pkl in tqdm(os.listdir(train_pyg_dir)):
        print(pkl)
        g = torch.load(os.path.join(train_pyg_dir,pkl))
        cluster = c0.fit(g.x)
        g['cluster_level1_'+str(i)] = torch.LongTensor(cluster.labels_).cpu()
        g['cluster_level2_'+str(i)] = -1 * torch.ones(len(g.x)).cpu().long()
        for ci in range(i):
            if ks[1] > len(g.x[g['cluster_level1_'+str(i)]==ci]):
                c_i = KMeans(n_clusters=len(g.x[g['cluster_level1_'+str(i)]==ci]), random_state=0)
                cluster = c_i.fit(g.x[g['cluster_level1_'+str(i)]==ci])
            else:
                cluster = c.fit(g.x[g['cluster_level1_'+str(i)]==ci])
            g['cluster_level2_'+str(i)][g['cluster_level1_'+str(i)]==ci] = torch.LongTensor(cluster.labels_).cpu()
        torch.save(g,os.path.join(train_pyg_dir,pkl))
        print(np.unique(cluster.labels_,return_counts=True))