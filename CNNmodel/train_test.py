print('启动正常')
import numpy as np
from net1d import *
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset,DataLoader,ConcatDataset
from sklearn.model_selection import train_test_split
import sys
sys.path.append('/home/linzenghui/ECG_code/HeartRateVariability_220217')
import FrequencyDomain as fd
import TimeDomain as td
import NonLinear as nl
from common import *
from Rpeaks import *

all_dataset=['wsc','shhs1','shhs2','cfs','numom2b1','numom2b2']
use_dataset=['wsc','shhs1','shhs2','cfs']

def r_peaks(sig,fs=200):
    peaks=simple_qrs_detector(sig,fs=fs)
    rpos = R_Wave_finetune(sig, peaks)
    return rpos

def basic_screen(sig,rpos,fs=200):
    if len(rpos)<150:
        return (False,'峰值太少')
        # Amplitude less than 3mV
        # ampl = np.abs(np.max(sig) - np.min(sig))
        #if ampl > 3:
        #    return False
    sig_len=len(sig)
    tmp_sig = np.abs(sig)
    tmp_sig = tmp_sig[tmp_sig > 0.1]
    if len(tmp_sig) < 5:
        return (False,'电压值过低')
    if rpos[0] > fs*5 or rpos[-1] < (sig_len-fs*5):
        return (False,'前方或后方有空缺')
    rr_intervals = np.diff(rpos)
    maxRR = np.max(rr_intervals)
    meanRR = np.mean(rr_intervals)
    if maxRR > meanRR * 3:
        return (False,'rri max值过大')
    if maxRR>fs*5:
        return (False,'有超过5秒的空白')
    return (True,'pass')

def cal_corr_coeff_lst(sig, rpos):
        beat_seg = []
        for idx, r_p in enumerate(rpos):
            if r_p > 200 and (len(sig) - r_p) > 200:
                tmp_seg = sig[r_p - 200:r_p + 200]
                beat_seg.append(tmp_seg)
        beat_seg = np.array(beat_seg)
        template_qrs = np.mean(beat_seg, axis=0)
        template_qrs = template_qrs - np.mean(template_qrs)
        coeff_lst = []
        for seg in beat_seg:
            seg = seg - np.mean(seg)
            coeff = np.corrcoef(seg, template_qrs)[0, 1]
            coeff_lst.append(coeff)
        return coeff_lst

def sqi(sig, rpos,control):
        preRes = basic_screen(sig,rpos)

        if preRes[0]:
            rpos = rpos
        else:
            return (preRes[1], float(0))

        coeff_lst = cal_corr_coeff_lst(sig, rpos)
        # template_nums = self.check_coeff(coeff_lst)
        coeff = float(np.mean(coeff_lst))

        if coeff > control:
            return (True, coeff)
        else:
            return (False, coeff)

def normalize_for_data(data):
    data=data/(np.max(data)-np.min(data))
    return data

def normalize_for_database(database):
    for idx in range(len(database)):
        database[idx,6:]=normalize_for_data(database[idx,6:])
    return database    

def prepare_data(dataset_used,use_normal=True,use_control=True,control_sqi=0.6):
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
    train_base=np.vstack(train_dataset_list)
    test_base=np.vstack(test_dataset_list)
    if use_control:
        print(f'开始质量控制{control_sqi}')
        pass_train=[sqi(train_base[idx,6:],r_peaks(train_base[idx,6:]),control=control_sqi)[0] for idx in range(len(train_base))]
        train_base=train_base[pass_train]
        pass_test=[sqi(test_base[idx,6:],r_peaks(test_base[idx,6:]),control=control_sqi)[0] for idx in range(len(test_base))]
        test_base=test_base[pass_test]
        print('质量控制结束')
    print(f'训练集:{len(train_base)},测试集{len(test_base)}')
    if use_normal:
        print('数据正则化开始')
        train_base=normalize_for_database(train_base)
        test_base=normalize_for_database(test_base)
        print('数据正则化完成')
    return(train_base,test_base)

train_data,test_data=prepare_data(dataset_used=use_dataset,use_control=True,control_sqi=0.6)

def cut_data(data,window_size=30*200,step=30*200,datasetnumber=6):
    database=np.zeros(shape=(int(data.shape[0]*((data.shape[1]-6-window_size)/step+1)),6+window_size))
    database[:,0]=datasetnumber
    count=0
    for idx in range(data.shape[0]):
        for start in range(6,data.shape[1],step):
            database[count,1]=data[idx,1]
            database[count,2]=data[idx,2]
            database[count,3]=data[idx,3]
            database[count,4]=data[idx,4]
            database[count,5]=data[idx,5]
            database[count,6:]=data[idx,start:start+window_size]
            count+=1
    assert count==database.shape[0]
    return database

class Psg_dataset(Dataset):
    def __init__(self,state,use_cut) :
        if state=='train':
            self.data_all=train_data
            if use_cut:
                self.data_all=cut_data(train_data)
                print('train_use_cut')
        if state=='test':
            self.data_all=test_data
            if use_cut:
                self.data_all=cut_data(test_data)
                print('test_use_cut')
    
    def __getitem__(self, index) :
            return self.data_all[index,6:],self.data_all[index,2]

    def __len__(self):
        return len(self.data_all)

psg_data=Psg_dataset(state='train',use_cut=True)
train_iter=DataLoader(psg_data,batch_size=64,shuffle=True)

model=Net1D(in_channels=1,base_filters=256,ratio=1,
          filter_list=[256,512,512,256,128,32],m_blocks_list=[3,3,4,4,3,1],
          kernel_size=8,stride=2,
          n_classes=1,use_bn=True,use_do=True,verbose=False,groups_width=16)
device_str = torch.device('cuda:5')
model.to(device_str)
lossfun=nn.BCELoss()##自动求平均的
updater=torch.optim.Adam(model.parameters(),lr=1e-3)


train_loss=[]
for epoch in range(201):
    model.train()
    epochloss=[]
    for train_x,train_y in train_iter:
        train_x=torch.tensor(np.expand_dims(train_x,1)).to(device_str,torch.float32)
        train_y=torch.tensor(np.expand_dims(train_y,1)).to(torch.float32).to(device_str)
        y_hat=torch.sigmoid(model(train_x))
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

# torch.save(model,'/data/0shared/linzenghui/ECG_data/public_dataset/ECG_model/model_045')
# print('模型已保存至model_045')
# print('''model=Net1D(in_channels=1,base_filters=256,ratio=1,
#           filter_list=[256,512,512,256,128,32],m_blocks_list=[3,3,4,4,3,1],
#           kernel_size=8,stride=2,
#           n_classes=2,use_bn=True,use_do=True,verbose=False,groups_width=16)
# device = torch.device('cuda:8')
# model.to(device)
# lossfun=nn.CrossEntropyLoss()##自动求平均的
# updater=torch.optim.Adam(model.parameters(),lr=1e-3)''')


test_psg_data=Psg_dataset(state='test',use_cut=True)
test_iter=DataLoader(test_psg_data,batch_size=64,shuffle=False)
model.eval()
pro=torch.tensor([]).to(device_str)
true_y=torch.tensor([])
with torch.no_grad():
    for val_x,val_y in test_iter:
        val_x=torch.tensor(np.expand_dims(val_x,1)).to(device_str,torch.float32)
        y_hat=torch.sigmoid(model(val_x))
        pro=torch.cat((pro,y_hat))
        true_y=torch.cat((true_y,val_y))
pre=pro.to('cpu')
value=roc_auc_score(true_y,pre)
print(f"auc:{value}")