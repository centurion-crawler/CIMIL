import os
import joblib
import torch
from torch.utils.data import Dataset,Subset
from torch_geometric.loader import DataLoader

class CM_Dataset(Dataset):
    def __init__(self,fold_index,train_dir='/data12/zzf/MIL/CM16_256/train_patches/pkl_bak',test_dir='/data12/zzf/MIL/CM16_256/test_patches/pkl_bak'):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.test = 'test' in fold_index
        folds_file = joblib.load('/data12/zzf/MIL/CM16/data/fold_split/fold_split.pkl')
        if 'all' in fold_index:
            self.graphs = list(set(folds_file[fold_index.replace('all','train')])|set(folds_file[fold_index.replace('all','val')])|set(folds_file[fold_index.replace('all','test')]))
        else:
            self.graphs = folds_file[fold_index]
        self.num_features = 1024
        self.num_classes=2

    def __getitem__(self,index):
        if 'test' in self.graphs[index]:
            graph_path = os.path.join(self.test_dir,self.graphs[index])
        else:
            graph_path = os.path.join(self.train_dir,self.graphs[index])
        graph = torch.load(graph_path)
        return graph, graph_path
    
    def __len__(self):
        return len(self.graphs)
        
            
def get_dataloader(index=0,seed=0):
    train_set=CM_Dataset(fold_index="train_{}".format(index))
    val_set=CM_Dataset(fold_index="val_{}".format(index))
    test_set=CM_Dataset(fold_index="test_{}".format(index))
    all_set=CM_Dataset(fold_index="all_{}".format(index))
    dataloader={}
    dataloader["train"]=DataLoader(train_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["all"]=DataLoader(all_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["val"]=DataLoader(val_set,batch_size=1,num_workers=0,drop_last=False,shuffle=True)
    dataloader["test"]=DataLoader(test_set,batch_size=1,num_workers=0,drop_last=False)

    return dataloader