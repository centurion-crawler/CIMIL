import os
import sys
import torch
import yaml
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.autograd import Variable

from sklearn.metrics import classification_report,accuracy_score,roc_curve,auc,accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from collections import Counter
from load_dataset import get_dataloader
from easydict import EasyDict

config_path = sys.argv[1]
with open(config_path, 'r') as f:
    config = EasyDict(yaml.safe_load(f))

def log_print(f_name,str_text):
    with open(f_name,'a') as f:
        f.write(str(str_text)+'\n')

def save_model(model,val_acc,epoch,fold,pathname,f_name):
    log_print(f_name,'saving....')
    s_model=model.to('cpu')
    state = {
        'net': s_model.state_dict(),
        'epoch': epoch,
        'acc': val_acc
    }
    torch.save(state,pathname)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)

class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class CLAM_MB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        super(CLAM_MB, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict


def epoch_test(dataloader_, criterion, milnet, fc):
    bag_labels = []
    bag_preds = []
    bag_preds_score = []
    epoch_loss = 0.0
    with torch.no_grad():
        for data,graph_path in dataloader_:
            graph = data.to(device)
            label = graph.bag_label.long()
            fea_x = F.normalize(fc(graph.x),dim=-1).to(device)
            datax = fea_x
            # datax = torch.cat((F.normalize(graph.x,dim=-1),fea_x),dim=-1)
            logits, Y_prob, Y_hat, A_raw, results_dict = milnet(datax) 

            bag_pred = Y_prob

            loss_bag = criterion(bag_pred, label)

            epoch_loss = epoch_loss + loss_bag.item()

            bag_labels.append(label.item())
            bag_preds.append(bag_pred[0,1].item())
            # bag_preds_score.append(bag_pred.item())

    epoch_loss = epoch_loss / len(dataloader_)
    return epoch_loss, bag_labels,bag_preds       

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def five_scores(bag_labels, bag_predictions):
    print(bag_labels,bag_predictions)
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    # precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
    acc=accuracy_score(bag_labels, bag_predictions)
    # return accuracy, auc_value, precision, recall, fscore
    return acc, auc_value
    
device = torch.device("cuda:0")
epoch = config.Warmup.epoch
early_stop = config.Warmup.early_stop
batch_size= config.Warmup.batch_size

os.makedirs('checkpoint',exist_ok=True)
os.makedirs('logs',exist_ok=True)


fi = config.fi
os.makedirs('logs',exist_ok=True)
f_name = 'logs/Warmup___WSI_fea_CLAM_MB-MIL-256_epoch_{}_early_stop_{}_batch_size_{}_fold_{}.log'.format(config.Warmup.epoch,config.Warmup.early_stop,config.Warmup.batch_size,fi)

log_print(f_name,'epoch_{}_early_stop_{}_batch_size_{}_fold_{}.log'.format(config.Warmup.epoch,config.Warmup.early_stop,config.Warmup.batch_size,fi))
log_print(f_name,"=======================FOLD {}=======================".format(fi+1))


milnet = CLAM_MB().to(device)

dataloader = get_dataloader(index=0)
criterion=torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(milnet.parameters(), lr=config.Warmup.lr, betas=(0.5, 0.9), weight_decay=config.Warmup.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, 0)

best_val_acc = 0.0
early_stop_count = 0

for ei in range(0,epoch):
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
        logits, Y_prob, Y_hat, A_raw, results_dict = milnet(graph) 

        bag_pred = Y_prob

        if labels is None:
            labels = label
            bag_preds = bag_pred
        else:
            labels=torch.cat([labels,label])
            bag_preds=torch.cat([bag_preds,bag_pred])
            
        if m%batch_size==0 or m==len(dataloader["train"]):
            
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
    _,train_bag_labels,train_bag_preds = epoch_test(dataloader["train"],criterion, milnet)
    train_acc,train_auc = five_scores(train_bag_labels, train_bag_preds)
    val_loss,val_bag_labels,val_bag_preds = epoch_test(dataloader["val"],criterion, milnet)
    val_acc,val_auc = five_scores(val_bag_labels, val_bag_preds)
    test_loss,test_bag_labels,test_bag_preds = epoch_test(dataloader["test"],criterion, milnet)
    test_acc,test_auc = five_scores(test_bag_labels, test_bag_preds)

    if ei > epoch*0.2:
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


