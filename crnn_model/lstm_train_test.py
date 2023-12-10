import torch
from torch import nn
from cnn_lstm import cnn_lstm1d
from torch.utils.data import Dataset,DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

all_dataset=['wsc','shhs1','shhs2','cfs','numom2b1','numom2b2']
use_dataset=['numom2b2']

def prepare_data(dataset_used):
    train_dataset_list=[]
    test_dataset_list=[]
    if 'wsc' in dataset_used:
        tmp_train_data,tmp_test_data=train_test_split(np.load('/data/0shared/linzenghui/ECG_data/public_dataset/wsc/analysis_extract/wsc.npy'),
                                                  test_size=0.2,random_state=100,shuffle=False)
        train_dataset_list.append(tmp_train_data)
        test_dataset_list.append(tmp_test_data)
        print('wsc loaded down')
    if 'shhs1' in dataset_used:
        tmp_train_data,tmp_test_data=train_test_split((np.load('/data/0shared/linzenghui/ECG_data/public_dataset/shhs/analysis_extract/shhs1.npy')),
                                                          test_size=0.2,random_state=100,shuffle=False)
        train_dataset_list.append(tmp_train_data)
        test_dataset_list.append(tmp_test_data)
        print('shhs1 loaded down')
    if 'shhs2' in dataset_used:
        tmp_train_data,tmp_test_data=train_test_split((np.load('/data/0shared/linzenghui/ECG_data/public_dataset/shhs/analysis_extract/shhs2.npy')),
                                                          test_size=0.2,random_state=100,shuffle=False)
        train_dataset_list.append(tmp_train_data)
        test_dataset_list.append(tmp_test_data)
        print('shhs2 loaded down')
    if 'cfs' in dataset_used:
        tmp_train_data,tmp_test_data=train_test_split((np.load('/data/0shared/linzenghui/ECG_data/public_dataset/cfs/analysis_extract/cfs.npy')),
                                                          test_size=0.2,random_state=100,shuffle=False)
        train_dataset_list.append(tmp_train_data)
        test_dataset_list.append(tmp_test_data)
        print('cfs loaded down')
    if 'numom2b1' in dataset_used:
        tmp_train_data,tmp_test_data=train_test_split((np.load('/data/0shared/linzenghui/ECG_data/public_dataset/numom2b/analysis_ectract/numom2b1.npy')),
                                                          test_size=0.2,random_state=100,shuffle=False)
        train_dataset_list.append(tmp_train_data)
        test_dataset_list.append(tmp_test_data)
        print('numom2b1 loaded down')
    if 'numom2b2' in dataset_used:
        tmp_train_data,tmp_test_data=train_test_split((np.load('/data/0shared/linzenghui/ECG_data/public_dataset/numom2b/analysis_ectract/numom2b2.npy')),
                             test_size=0.2,random_state=100,shuffle=False)
        train_dataset_list.append(tmp_train_data)
        test_dataset_list.append(tmp_test_data)
        print('numom2b2 loaded down')
    return(np.vstack(train_dataset_list),np.vstack(test_dataset_list))

train_data,test_data=prepare_data(dataset_used=all_dataset)

class Psg_dataset(Dataset):
    def __init__(self,state) :
        if state=='train':
            self.data_all=train_data
        if state=='test':
            self.data_all=test_data
    
    def __getitem__(self, index) :
            return self.data_all[index,3:60003],self.data_all[index,2]

    def __len__(self):
        return len(self.data_all)
psg_data=Psg_dataset(state='train')

device='cuda:3'
train_iter=DataLoader(psg_data,batch_size=8,shuffle=False)
model=cnn_lstm1d(in_channels=1,out_channels=16,n_len_seg=2000,n_classes=2,verbose=False)
model.to(device)
lossfun=nn.CrossEntropyLoss()##自动求平均的
updater=torch.optim.Adam(model.parameters(),lr=1e-3)

train_loss=[]
for epoch in range(70):
    model.train()
    epochloss=[]
    for train_x,train_y in train_iter:
        train_x=torch.tensor(np.expand_dims(train_x,1)).to(device,torch.float32)
        train_y=train_y.to(torch.long).to(device)
        y_hat=model(train_x)
        updater.zero_grad()
        loss=lossfun(y_hat,train_y)
        epochloss.append(loss)
        loss.backward()
        updater.step()
    ls=sum(epochloss)/len(epochloss)
    # print(f"epoch{epoch},loss:{ls}")
    train_loss.append(ls)
    if epoch%10==0:
        print(f"epoch {epoch} down")
        print(f"epoch{epoch},loss:{ls}")

torch.save(model,'/data/0shared/linzenghui/ECG_data/public_dataset/ECG_model/model_036')
print('模型已保存至model_036')
print('''device='cuda:3'
train_iter=DataLoader(psg_data,batch_size=8,shuffle=False)
model=cnn_lstm1d(in_channels=1,out_channels=16,n_len_seg=2000,n_classes=2,verbose=True)
model.to(device)
lossfun=nn.CrossEntropyLoss()##自动求平均的
updater=torch.optim.Adam(model.parameters(),lr=1e-3)''')

test_psg_data=Psg_dataset(state='test')
test_iter=DataLoader(test_psg_data,batch_size=8,shuffle=False)
model.eval()
pro=torch.tensor([]).to(device)
true_y=torch.tensor([])
with torch.no_grad():
    for val_x,val_y in test_iter:
        val_x=torch.tensor(np.expand_dims(val_x,1)).to(device,torch.float32)
        y_hat=torch.argmax(model(val_x),1)
        pro=torch.cat((pro,y_hat))
        true_y=torch.cat((true_y,val_y))
pre=pro.to('cpu')
value=roc_auc_score(true_y,pre)
print(f"auc:{value}")