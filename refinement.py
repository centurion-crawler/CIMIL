import os
import sys
import torch
import joblib 
import argparse
import numpy as np
import torch.nn.functional as F
from torch import Tensor
import pandas as pd 
from torch_geometric.data import Data
from torch.autograd import Variable
from torch.nn import Linear,Dropout,LayerNorm
from sklearn.metrics import classification_report,accuracy_score,roc_curve,auc,balanced_accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from collections import Counter
from models import *
from load_dataset import get_dataloader
import torch.nn as nn
from typing import Optional, Sequence

config_path = sys.argv[1]
# config_path = "/data12/zzf/MIL/AAAI_github/CIMIL/config/config.yml"
with open(config_path, 'r') as f:
    config = EasyDict(yaml.safe_load(f))
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

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

def epoch_test(dataloader_, criterion, milnet, prj):
    bag_labels = []
    bag_preds = []
    bag_preds_score = []
    epoch_loss = 0.0
    with torch.no_grad():
        for data,graph_path in dataloader_:
            graph = data.to(device)
            label = graph.bag_label.long()
            fea_x = F.normalize(prj(graph.x),dim=-1).to(device)
            datax = torch.cat((F.normalize(graph.x,dim=-1),fea_x),dim=-1)
            logits, Y_prob, Y_hat, A_raw, results_dict = milnet(datax) 
            bag_pred = Y_prob
            loss_bag = criterion(bag_pred, label)
            epoch_loss = epoch_loss + loss_bag.item()
            bag_labels.append(label.item())
            bag_preds.append(bag_pred[0,1].item())

    epoch_loss = epoch_loss / len(dataloader_)
    return epoch_loss, bag_labels,bag_preds   

def five_scores(bag_labels, bag_predictions):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    acc=accuracy_score(bag_labels, bag_predictions)
    return acc, auc_value

device = torch.device("cuda:0")
epoch = config.Refine.epoch
early_stop = config.Refine.early_stop
batch_size = config.Refine.batch_size
fi = config.fi
mu = config.mu

os.makedirs('checkpoint', exist_ok=True)
os.makedirs('logs', exist_ok=True)

f_name = 'logs/Tuning_refine_CLAM_MB-MIL-256_epoch_{}_early_stop_{}_batch_size_{}_fold_{}.log'.format(epoch,early_stop,batch_size,fi)
if fi == 0:
    f_ = open(f_name, 'w')
    f_.truncate()
log_print(f_name,'epoch_{}_early_stop_{}_batch_size_{}_fold_{}.log'.format(epoch,early_stop,batch_size,fi))
log_print(f_name,"=======================FOLD {}=======================".format(fi))

milnet = CLAM_MB().to(device)
prj = Projector().to(device)
prj.load_state_dict(torch.load(config.Refine.prj_weight)["net"])
prj.eval()

dataloader = get_dataloader(index=fi)
if config.Refine.creiterion == "cross-entropy":
    criterion = nn.CrossEntropyLoss().to(device)
elif config.Refine.creiterion == "focal":
    criterion = FocalLoss(alpha=torch.Tensor(config.Refine.focal_alpha)).to(device)
else:
    raise NotImplementedError
optimizer = torch.optim.Adam(milnet.parameters(), lr=config.Refine.lr, weight_decay=config.Refine.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, 0)

best_val_acc = 0.0
early_stop_count = 0

for ei in range(epoch):
    milnet.train()
    milnet.to(device)
    m = 0
    epoch_loss = 0.0
    batch_num = 0 

    labels = None
    bag_preds = None
    for data,graph_path in dataloader["train"]:
        graph = data.to(device)
        m+=1
        label = torch.LongTensor([graph.bag_label]).to(device)
        fea_x = F.normalize(prj(graph.x),dim=-1).to(device)
        datax = torch.cat((F.normalize(graph.x,dim=-1),fea_x),dim=-1)
        logits, Y_prob, Y_hat, A_raw, results_dict = milnet(datax) 

        bag_pred = Y_prob

        if labels is None:
            labels = label
            bag_preds = bag_pred
        else:
            labels=torch.cat([labels,label])
            bag_preds=torch.cat([bag_preds,bag_pred])
            
        if m%config.batch_size==0 or m==len(dataloader["train"]):
            
            loss_bag = criterion(bag_preds, labels)
            epoch_loss = epoch_loss + loss_bag.item()*batch_size
            optimizer.zero_grad()
            loss_bag.backward()
            optimizer.step()
            batch_num+=1
            labels = None
            bag_preds = None
        del graph
    loss_ei = epoch_loss/m
    log_print(f_name,"----------Epoch {}----------".format(ei+1))
    log_print(f_name,"lr:{} Train on {} samples, batch_num:{} loss: {:.4f}".format(optimizer.param_groups[0]['lr'],m,batch_num,loss_ei))
    
    milnet.eval()
    _,train_bag_labels,train_bag_preds = epoch_test(dataloader["train"],criterion, milnet, prj)
    train_acc,train_auc = five_scores(train_bag_labels, train_bag_preds)
    val_loss,val_bag_labels,val_bag_preds = epoch_test(dataloader["val"],criterion, milnet, prj)
    val_acc,val_auc = five_scores(val_bag_labels, val_bag_preds)
    test_loss,test_bag_labels,test_bag_preds = epoch_test(dataloader["test"],criterion, milnet, prj)
    test_acc,test_auc = five_scores(test_bag_labels, test_bag_preds)

    if ei > epoch*float(config.Refine.begin_to_save_epoch_rate):
        if val_acc>=best_val_acc:
            best_val_acc = val_acc
            model_path_name = 'checkpoint/'+f_name[5:-4]+'_'+str(fi)+'.pth'
            save_model(milnet,val_acc,ei,fi,model_path_name,f_name)
            early_stop_count=0
        else:
            early_stop_count+=1
        
        if early_stop_count>=early_stop:
            break
    scheduler.step()

    log_print(f_name,"train acc:{} auc:{} ".format(train_acc,train_auc))
    log_print(f_name,"val acc:{} auc:{}  loss:{}".format(val_acc,val_auc,val_loss))
    log_print(f_name,"test acc:{} auc:{} loss:{}".format(test_acc,test_auc,test_loss))
    
del milnet


