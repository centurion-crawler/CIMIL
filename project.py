import os
import torch
import joblib 
import sys
import argparse
import numpy as np
import torch.nn.functional as F
from torch import Tensor
import pandas as pd 
from sklearn.metrics import classification_report,accuracy_score,roc_curve,auc,balanced_accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from collections import Counter
from load_dataset import get_dataloader
import torch.nn as nn
import random 
import yaml
from easydict import EasyDict
from tqdm import tqdm
from typing import Optional, Sequence
from models import Projector
config_path = sys.argv[1]
with open(config_path, 'r') as f:
    config = EasyDict(yaml.safe_load(f))
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id


def log_print(f_name,str_text):
    with open(f_name,'a') as f:
        f.write(str(str_text)+'\n')

def gen_idx(indexes,ks,mu):
    # [5,4,3]; [10,5,4] 0.02 
    
    assert len(indexes)>0
    c = g['cluster_level1_'+str(ks[0])]==indexes[0]
    for li in range(1,len(indexes)):
        if li+1>2:
            c = c&(g[f'cluster_level{li+1}_{ks[li]}_{mu}']==indexes[li])
        else:
            c = c&(g[f'cluster_level{li+1}_{ks[li]}']==indexes[li])
    return c

def save_model(model,val_acc,epoch,fold,pathname,f_name):
    log_print(f_name,'saving....')
    s_model=model.to('cpu')
    state = {
        'net': s_model.state_dict(),
        'epoch': epoch,
        'acc': val_acc
    }
    torch.save(state,pathname)


class FocalLoss(nn.Module):
    
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 3.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]
        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        # log_p = F.softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss



def epoch_test(dataloader_, criterion, milnet, is_test=False, store=False):
    milnet.eval()
    
    aucs = []
    recalls = []
    precisions = []
    f1s = []

    epoch_loss = 0.0
    with torch.no_grad():
        epoch_loss = 0.0
        batch_num_epoch = 0 
        N_intances=0
        inst_labels_all = []
        inst_preds_all = []
        for data,graph_path in tqdm(dataloader_):
            inst_labels = []
            inst_preds = []
            N = data.x.shape[0]
            N_intances += N
            if N % config.batch_size==0:
                batch_nums = N//config.batch_size
            else:
                batch_nums = N//config.batch_size+1
            batch_num_epoch += batch_nums
            
            for bi in range(batch_nums):
                
                data_x = data.x[bi*config.batch_size:min((bi+1)*config.batch_size,N)].to(device)
                data_y = data.y_ins[bi*config.batch_size:min((bi+1)*config.batch_size,N)].to(device)

                inst_res_logits = milnet(data_x)
                loss_inst = criterion(inst_res_logits, data_y)
                
                inst_labels = inst_labels+list(data_y.detach().cpu().numpy())
                inst_preds = inst_preds+list(F.softmax(inst_res_logits,dim=-1)[:,1].detach().cpu().numpy())
                
                epoch_loss = epoch_loss + loss_inst.item()
            inst_labels_all = inst_labels_all + inst_labels
            inst_preds_all = inst_preds_all + inst_preds

            if store: # for FROC
                if is_test:
                    res = torch.cat((torch.Tensor(inst_preds).unsqueeze(1),data.pos),dim=1)
                    csv_data = res.cpu().numpy()
                    csv_Froc=pd.DataFrame(csv_data, columns=['Confidence','X coordinate','Y coordinate'])
                    csv_Froc.to_csv(os.path.join(config.Projector.res_test,os.path.basename(graph_path[0])[:-4]+'.csv'),index=0)
                else:
                    res = torch.cat((torch.Tensor(inst_preds).unsqueeze(1),data.pos),dim=1)
                    csv_data = res.cpu().numpy()
                    csv_Froc=pd.DataFrame(csv_data, columns=['Confidence','X coordinate','Y coordinate'])
                    csv_Froc.to_csv(os.path.join(config.Projector.res_train,os.path.basename(graph_path[0])[:-4]+'.csv'),index=0)

        acc,auc_,precision_,recall_,f1_ = five_scores(inst_labels_all,inst_preds_all)
        aucs.append(auc_)
        recalls.append(recall_)
        precisions.append(precision_)
        f1s.append(f1_)

    epoch_loss = epoch_loss / batch_num_epoch
    return epoch_loss, acc,np.mean(aucs), np.mean(precisions), np.mean(recalls), np.mean(f1s), len(aucs)          

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def five_scores(bag_labels, bag_predictions):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    # this_class_label[this_class_label>=threshold_optimal] = 1
    # this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='marco')
    acc= balanced_accuracy_score(bag_labels, bag_predictions)
    return acc, auc_value, precision, recall, fscore

device = torch.device("cuda:0")
epoch = config.Projector.epoch
early_stop = config.Projector.early_stop
batch_size = config.Projector.batch_size
mu = config.mu

os.makedirs('checkpoint', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs(config.Projector.res_train,exit_ok=True)
os.makedirs(config.Projector.res_test,exit_ok=True)

fi = config.fi
f_name = os.path.join('logs','Tuning_Projector_epoch_{}_early_stop_{}_batch_size_{}_fold_{}.log'.format(epoch,early_stop,batch_size,fi))
if fi==0:
    f = open(f_name,'w')
    f.truncate()
log_print(f_name,'epoch_{}_early_stop_{}_batch_size_{}_fold_{}.log'.format(epoch,early_stop,batch_size,fi))
log_print(f_name,"=======================FOLD {}=======================".format(fi))
milnet = Projector(n_classes=config.Projector.classes).to(device)
dataloader = get_dataloader(index=fi)

if config.Projector.creiterion == "cross-entropy":
    criterion = nn.CrossEntropyLoss().to(device)
elif config.Projector.creiterion == "focal":
    criterion = FocalLoss(alpha=torch.Tensor(config.Projector.focal_alpha))
else:
    raise NotImplementedError
optimizer = torch.optim.Adam(milnet.parameters(), lr=config.Projector.lr, weight_decay=config.Projector.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, 0)

best_val_auc = 0.0
early_top_count = 0
pseudo_idx = joblib.load(config.Projector.pseudo_idx_path)
# pseudo_idx = joblib.load(f'./Pseudo_label_{mu}_c_{c_num}_f_{fi}.pkl')

# P_train_data_x
p_train_data_x = torch.empty(0)
for pos_list in pseudo_idx['pos']:
    g = torch.load(pos_list[0])
    cond = gen_idx(pos_list[1:],config.cluster.ks,mu)
    p_train_data_x = torch.cat((g.x[cond].cpu(),p_train_data_x))
p_train_data_y = torch.ones(len(p_train_data_x)).long()

# N_train_data_x
n_train_data_x = torch.empty(0)
for neg_list in pseudo_idx['neg']:
    g = torch.load(neg_list[0])
    cond = gen_idx(neg_list[1:],config.cluster.ks,mu)
    n_train_data_x = torch.cat((g.x[cond].cpu(),n_train_data_x))
n_train_data_y = torch.zeros(len(n_train_data_x)).long()
train_x = torch.cat((p_train_data_x,n_train_data_x))
train_y = torch.cat((p_train_data_y,n_train_data_y))

for ei in range(0,epoch):
    Index = torch.LongTensor(random.sample(list(np.arange(len(train_y))),len(train_y)))
    train_data_x = train_x[Index]
    train_data_y = train_y[Index]
    milnet.train()
    milnet.to(device)
    epoch_loss = 0.0
    labels = None
    bag_preds = None
    N = train_data_x.shape[0]
        
    if N % config.batch_size==0:
        batch_nums = N//config.batch_size
    else:
        batch_nums = N//config.batch_size+1
        
    aucs = []
    recalls = []
    precisions = []
    f1s = []
    for bi in range(batch_nums):
        inst_labels = []
        inst_preds = []
        data_x = train_data_x[bi*config.batch_size:min((bi+1)*config.batch_size,N)].to(device)
        data_y = train_data_y[bi*config.batch_size:min((bi+1)*config.batch_size,N)].to(device)
        inst_res_logits = milnet(data_x) 
        loss_inst = criterion(inst_res_logits, data_y)
        optimizer.zero_grad()
        loss_inst.backward()
        optimizer.step()
        epoch_loss = epoch_loss + loss_inst.item()
        inst_labels = list(data_y.detach().cpu().numpy())
        inst_preds = list(F.softmax(inst_res_logits,dim=-1)[:,1].detach().cpu().numpy())
        acc,auc_,precision_,recall_,f1_ = five_scores(inst_labels,inst_preds)
        aucs.append(auc_)
        precisions.append(precision_)
        recalls.append(recall_)
        f1s.append(f1_)
        
    loss_ei = epoch_loss/batch_nums
    log_print(f_name,"----------Epoch {}----------".format(ei+1))
    log_print(f_name,"Train on {} instances,batch_num:{} loss: {:.4f} auc: {:.4f} ".format(N,batch_nums,loss_ei,np.mean(aucs)))
    
    milnet.eval()
    train_loss,train_acc,train_auc,train_precision,train_recall,train_f1,train_num = epoch_test(dataloader["train"],criterion, milnet)   
    val_loss,val_acc,val_auc,val_precision,val_recall,val_f1,val_num = epoch_test(dataloader["val"],criterion, milnet)
    test_loss,test_acc,test_auc,test_precision,test_recall,test_f1,test_num = epoch_test(dataloader["test"],criterion, milnet, is_test=True)

    if ei > epoch*float(config.Projector.begin_to_save_epoch_rate):
        if val_auc>=best_val_auc:
            best_val_auc = val_auc
            model_path_name = os.path.join('checkpoint',f_name[5:-4]+'_'+str(fi)+'.pth')
            save_model(milnet, val_auc, ei, fi, model_path_name, f_name)
            train_loss,train_acc,train_auc,train_precision,train_recall,train_f1,train_num = epoch_test(dataloader["train"],criterion, milnet, store=True)   
            val_loss,val_acc,val_auc,val_precision,val_recall,val_f1,val_num = epoch_test(dataloader["val"],criterion, milnet, store=True)
            test_loss,test_acc,test_auc,test_precision,test_recall,test_f1,test_num = epoch_test(dataloader["test"],criterion, milnet, is_test=True, store=True)
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count>=early_stop:
            break
    scheduler.step()
    log_print(f_name,"train acc:{} auc:{} precision:{}, recall:{}, f1:{}, loss:{}, number:{}".format(train_acc, train_auc, train_precision, train_recall, train_f1, train_loss, train_num))
    log_print(f_name,"val acc:{} auc:{} precision:{}, recall:{}, f1:{}, loss:{}, number:{}".format(val_acc, val_auc, val_precision, val_recall, val_f1, val_loss, val_num))
    log_print(f_name,"test acc:{} auc:{} precision:{}, recall:{}, f1:{}, loss:{}, number:{}".format(test_acc, test_auc, test_precision, test_recall, test_f1, test_loss, test_num))
del milnet