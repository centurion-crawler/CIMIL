import os
import sys 
import math
import torch
import joblib
import random 
import yaml
from models import CLAM_MB
import numpy as np 
from tqdm import tqdm 
from easydict import EasyDict
from sklearn.cluster import KMeans
from load_dataset import get_dataloader
from sklearn.metrics import classification_report,accuracy_score,roc_curve,auc,accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, confusion_matrix
 

config_path = sys.argv[1]
with open(config_path, 'r') as f:
    config = EasyDict(yaml.safe_load(f))
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

def epoch_test(dataloader_, criterion, milnet, neg_set, clu_num):
    milnet.eval()
    bag_labels = []
    bag_preds = []
    bag_preds_score = []
    epoch_loss = 0.0
    neg_set_k = list(neg_set.keys())
    res = {}
    with torch.no_grad():
        for data,graph_path in tqdm(dataloader_):
            res[graph_path[0]] = {}
            graph = data 
            graph[f'cluster_level3_{clu_num}'] = -1 * torch.ones(len(graph.x)).cpu().long()
            label = graph.bag_label.long()
            datax_ori = graph.x
            datax_ori = datax_ori.to(device)
            s,bag_pred,_,_,_ = milnet(datax_ori) 
            res[graph_path[0]][-1]=bag_pred[0,1].item()
            res[graph_path[0]]['y'] = label
            res[graph_path[0]]['y_inst'] = graph.y_ins
            if bag_pred[0,1].item()<1/2:
                continue
            exec(f"res[graph_path[0]]['cluster'] = graph.cluster_{clu_num}")
            exec(f"g_c_ = graph.cluster_level1_{clu_num}")
            exec(f"l_c_ = graph.cluster_level2_{clu_num}")
            g_c = locals()['g_c_']
            l_c = locals()['l_c_']

            for p in range(clu_num):
                res[graph_path[0]][p] = []
                datax = graph.x
                sets_M = [neg_set[random.choice(neg_set_k)][random.randint(0,clu_num-1)] for t in range(clu_num-1)]
                g_c = locals()['g_c_']
                datax_p = torch.cat((datax[g_c==p],torch.cat(sets_M)))
                datax_p = datax_p.to(device)
                s,bag_pred,_,_,_ = milnet(datax_p)
                res[graph_path[0]][p].append(bag_pred[0,1].item())
            
            p_set = []
            p2_set=[] 
            datax = graph.x
            for p in range(clu_num):
                if (res[graph_path[0]][-1]-res[graph_path[0]][p][0]<mu*10):
                    p_set.append(p)
            pos_idx = None 
            for p in p_set:
                if pos_idx is None:
                    pos_idx = (g_c==p)
                else:
                    pos_idx = (g_c==p) | pos_idx
            if pos_idx is None:
                continue
            others = datax[~pos_idx]
            set_M_List = [neg_set[random.choice(neg_set_k)][random.randint(0,clu_num-1)] for t in range(max(1,math.ceil(len(p_set))))]
            sets_M = torch.cat(tuple(set_M_List))
            for p in p_set:
                res[graph_path[0]][p].append({})
                for i in range(config.cluster.ks[1]):
                    data_ = torch.cat((datax[(g_c==p)&(l_c==i)],sets_M,others))
                    data_ = data_.to(device)
                    s,bag_pred,_,_,_ = milnet(data_)
                    res[graph_path[0]][p][1][i]=bag_pred[0,1].item()
                if not min([res[graph_path[0]][-1]-res[graph_path[0]][p][1][i]<mu for i in range(config.cluster.ks[1])]):
                    for j in range(config.cluster.ks[1]):
                        if res[graph_path[0]][-1]-res[graph_path[0]][p][1][j]<mu:
                            km = KMeans(n_clusters=min(config.cluster.ks[2],len(datax[(g_c==p)&(l_c==j)])), random_state=config.cluster.seed)
                            clu_km = km.fit(datax[(g_c==p)&(l_c==j)])
                            graph[f'cluster_level3_{cnum}_{mu}'][(graph[f'cluster_level1_{cnum}']==p)&(graph[f'cluster_level2_{cnum}']==j)] = torch.LongTensor(clu_km.labels_).cpu()
                            p2_set.append([p,j])
            pos_idx = None 
            for p in p2_set:
                if not isinstance (p,list):
                    continue
                if pos_idx is None:
                    pos_idx = (g_c==p[0]) & (l_c==p[1])
                else:
                    pos_idx = ((g_c==p[0]) & (l_c==p[1])) | pos_idx
            if pos_idx is None:
                continue
            datax = graph.x
            others = datax[~pos_idx]
            print('level3 len pos:',len(datax[pos_idx]))
            set_M_List = [neg_set[random.choice(neg_set_k)][random.randint(0,clu_num-1)] for t in range(max(1,math.ceil(len(p2_set)/2)))]
            sets_M = torch.cat(tuple(set_M_List))
            for p in p2_set:
                res[graph_path[0]][p[0]][1][p[1]] = [res[graph_path[0]][p[0]][1][p[1]]]
                res[graph_path[0]][p[0]][1][p[1]].append({})
                for i in range(config.cluster.ks[2]):
                    pos_data = datax[(g_c==p[0])&(l_c==p[1])&(graph[f'cluster_{clu_num}_level3']==i)]
                    print('pos_data:',len(pos_data))
                    data_ = torch.cat((pos_data,sets_M,others))
                    data_ = data_.to(device)
                    s,bag_pred,_,_,_ = milnet(data_)
                    res[graph_path[0]][p[0]][1][p[1]][1][i]=bag_pred[0,1].item()
            g = torch.load(graph_path[0])
            g[f'cluster_level3_{cnum}_{mu}'] = graph[f'cluster_level3_{cnum}_{mu}']
            torch.save(g,graph_path[0])
    return res

device = torch.device("cuda:0")
cnum = config.cluster.ks[0]
fi = config.fi
ckpt_path = config.HIS.ckpt_path
mu = config.mu
diff_threshold = config.HIS.diff_threshold

exec(f"neg_set_{cnum} = negative_set(dataloader['train'],c_num={cnum})")
dataloader = get_dataloader(index=fi)
milnet = CLAM_MB().to(device)
m_dict = torch.load(ckpt_path)
milnet.load_state_dict(m_dict['net'])
milnet.eval()
exec(f"res = epoch_test(dataloader['train'],criterion, milnet, neg_set_{cnum},{cnum})")
exec(f"joblib.dump(res,'CLAM_MB_test_res_{cnum}_fi_{fi}.pkl')")



def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]
def five_scores(bag_labels, bag_predictions):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    threshold_optimal=0.5
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
    c_m = confusion_matrix(bag_labels, bag_predictions)
    return auc_value,c_m


def negative_set(dataloader_,c_num):
    neg_set = {}
    si = 0
    for data,graph_path in dataloader_:
        if data.bag_label==1:
            continue
        neg_set[si] = {'ID':data.ID}
        for ci in range(c_num):
            exec(f"neg_set[si][ci] = data.x[data.cluster_level1_{c_num}==ci]")
        si += 1
    return neg_set

res= joblib.load(f'CLAM_MB_test_res_{cnum}_fi_{fi}.pkl')
# print(res)
dataloader = get_dataloader(index=fi)
neg_set = negative_set(dataloader['train'],cnum)


aucs = []
aucs_ = []
pos_plabel = []
neg_plabel = []

all_patch_nums = 0
for k in res.keys():
    max_set = []
    tmp_c_set = []
    tmp_set = []
    tmp2_set=[]
    g = torch.load(k)
    inst_label = g.y_ins
    inst_pred = inst_label.float()
    g = torch.load(k)
    inst_label = g.y_ins
    inst_pred = inst_label.float()
    if 'tumor' in k and res[k][-1]>config.HIS.pos_threshold and min([res[k][ci][0] for ci in range(cnum)])/sum([res[k][ci][0] for ci in range(cnum)]) < diff_threshold:
        change = 0
        min_dis = 1e9
        min_set = list((torch.topk(torch.Tensor([-res[k][i][0] for i in range(cnum)]),1)[1]).numpy())
        for i in range(cnum):
            if (res[k][-1]-res[k][i][0]<mu):
                change += 1
            else:
                continue
            if len(res[k][i])==2:
                for c in range(cnum//2):
                    if isinstance(res[k][i][1][c],list):
                        if ((config.cluster.ks[1]/config.cluster.ks[0]*res[k][i][1][c][0])>max(res[k][i][1][c][1].values())):
                            tmp2_set.append([i,c,res[k][i][1][c][0]])
                        else:
                            for c3 in range(config.cluster.ks[2]):
                                if (config.cluster.ks[1]/config.cluster.ks[0]*res[k][i][1][c][0]<res[k][i][1][c][1][c3]):
                                    tmp_set.append([i,c,c3,res[k][i][1][c][1][c3]])
                    elif (abs(res[k][-1]-res[k][i][1][c])<mu):
                        tmp2_set.append([i,c,res[k][i][1][c]])
            else:
                tmp_c_set.append([i,res[k][i][0]])
        for ti in tmp_c_set:
            max_set.append(ti)
        for ti in tmp_set:
            max_set.append(ti)

        for ti in tmp2_set:
            max_set.append(ti)
    else:
        continue
    inst_pred_t = []
    inst_label_t = []
    
    if len(max_set)==0 :
        continue
    for ms in max_set:
        if len(ms)==2:
            exec(f"inst_label_t = inst_label_t + list(inst_label[(g.cluster_level1_{cnum}==ms[0])].numpy())")
            exec(f"inst_pred_t = inst_pred_t + [ms[1]] * len(inst_label[(g.cluster_level1_{cnum}==ms[0])].numpy())")
        if len(ms)==3:
            exec(f"inst_label_t = inst_label_t + list(inst_label[(g.cluster_level1_{cnum}==ms[0]) & (g.cluster_level2_{cnum}==ms[1])].numpy())")
            exec(f"inst_pred_t = inst_pred_t + [ms[2]] * len(inst_label[(g.cluster_level1_{cnum}==ms[0]) & (g.cluster_level2_{cnum}==ms[1])].numpy())")
        elif len(ms)==4:
            exec(f"inst_label_t = inst_label_t + list(inst_label[(g.cluster_level1_{cnum}==ms[0]) & (g.cluster_level2_{cnum}==ms[1]) & (g['cluster_level3_{cnum}_{mu}']==ms[2])].numpy())")
            exec(f"inst_pred_t = inst_pred_t + [ms[3]] * len(inst_label[(g.cluster_level1_{cnum}==ms[0]) & (g.cluster_level2_{cnum}==ms[1]) & (g['cluster_level3_{cnum}_{mu}']==ms[2])].numpy())")
        
    for mis in min_set:
        exec(f"inst_label_t = inst_label_t + list(inst_label[(g.cluster_level1_{cnum}==mis)].numpy())")
        exec(f"inst_pred_t = inst_pred_t + [res[k][mis][0]] * len(inst_label[(g.cluster_level1_{cnum}==mis)].numpy())")
    
    if len(np.unique(inst_label_t))<2: # 伪标签只有一类也跳过
        print('skip only one class in Pseudo')
        continue

    if sum(inst_label_t)/len(inst_label_t)<config.HIS.P_low_bound: # 伪标签太少就跳过
        print('Too few positive pseudo labels')
        continue
    

    add_label_t = [0]*(len(inst_label_t)//2)
    add_pred_t = [0.0]*(len(inst_pred_t)//2)
    inst_pred_t = inst_pred_t + add_pred_t
    inst_label_t = inst_label_t + add_label_t

    print(k,'max_min_set:',max_set,min_set)

    auc,c_m = five_scores(inst_label_t,inst_pred_t)
    for ms in max_set:
        if len(ms)==2:
            pos_plabel.append([k,ms[0]])
        if len(ms)==3:
            pos_plabel.append([k,ms[0],ms[1]])
        elif len(ms)==4:
            pos_plabel.append([k,ms[0],ms[1],ms[2]])
    for mis in min_set:
        neg_plabel.append([k,mis])
        neg_plabel.append([os.path.join(config.cluster.train_dir,neg_set[random.choice(list(neg_set.keys()))]['ID'][0][:-3]+'.pkl'),random.randint(0,config.cluster.ks[0]-1)])
            

joblib.dump({'pos':pos_plabel,'neg':neg_plabel},f'./Pseudo_label_{mu}_c_{cnum}_f_{fi}.pkl')    

# 后验评估伪标签准确率时用
# aucs.append(auc*len(inst_label_t))
# aucs_.append(auc)
# all_patch_nums += len(inst_label_t)

# print('weighted AUC:',np.sum(aucs)/all_patch_nums)
# print('mean AUC:',np.mean(aucs_))
# print('ALL patches:',all_patch_nums)