import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
import os
import pytorch_lightning as pl
import json
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision import models
import time
import copy
import random
import gc
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter
import cv2
from pytorch_lightning.callbacks import ModelCheckpoint

resize = transforms.Resize((224,224))
totensor = transforms.ToTensor()
toPIL = transforms.ToPILImage()
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
##### load model  #########
class Pretrain_model(pl.LightningModule):
    def __init__(self,lr=0.0001,use_timm = True,loss_fun = nn.CrossEntropyLoss(),num_classes=5):
        super().__init__()
        self.backbone = None
        self.backbone = model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        self.backbone.fc = nn.Linear(2048,num_classes,bias=True)
        self.loss_fun = loss_fun
        self.lr = lr
        
    def forward(self,x):
        out = self.backbone(x)
        return out
    
    def training_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self(x)
        loss = self.loss_fun(y_hat,y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fun(y_hat, y)
        self.log("val_loss", loss)
        return torch.argmax(y_hat,dim = 1), y,y_hat
    
    def validation_epoch_end(self, validation_step_outputs):
        Y_hat = torch.tensor([])
        Y = torch.tensor([])
        Original_y_hat = torch.tensor([])
        for y_hat,y ,original_y_hat in validation_step_outputs:
            Y_hat = torch.cat((Y_hat,y_hat.cpu()))
            Y = torch.cat((Y,y.cpu()))
            Original_y_hat = torch.cat((Original_y_hat,original_y_hat.cpu()))
        loss = self.loss_fun(Original_y_hat, Y.long())
        f1 = f1_score(Y.detach().numpy(),Y_hat.detach().numpy(),average = 'micro')
        print('f1 score is :{}'.format(f1))
        print('val loss is :{}'.format(loss))
        self.log('f1_score',f1)
                
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99,verbose = True)
        return {'optimizer':optimizer,'lr_scheduler':scheduler,'interval':1}

class Original_dataset(Dataset):
    def __init__(self, imgs,path):
        self.imgs = imgs
        self.length = len(imgs)
        self.path = path
        
    def __len__(self):
        # RETURN SIZE OF DATASET
        return self.length

    def __getitem__(self, idx):
        # RETURN IMAGE AT GIVEN idx
        image = self.imgs[idx]
        image = Image.open(self.path +image)
        
        # augmentation
        image = resize(image)
        image = totensor(image)
        image = normalize(image)
        return image

def test_cnn(model,dl,val=False,times=5):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    global_outs = None
    torch.set_grad_enabled(False)
    m = nn.Softmax(dim=1)
    for i in range(times):
        outs = torch.tensor([])
        for imgs in tqdm(dl):
            if val:
                imgs,_ = imgs
            imgs = imgs.to(device)
            out = model(imgs)
            outs = torch.cat((outs,out.cpu()))
        outs = m(outs)
        if global_outs ==None:
            global_outs = outs
        else:
            global_outs += outs 
    return global_outs/times

test_path = 'car_data/test/images/'
test_images = os.listdir(test_path)
test_ds = Original_dataset(test_images,test_path)
test_dl = DataLoader(test_ds,batch_size=2,shuffle=False)

model_path = 'car_data/models/'
model_hw = Pretrain_model.load_from_checkpoint(model_path + 'model_hw.ckpt',num_classes =2)
model_wx = Pretrain_model.load_from_checkpoint(model_path + 'model_wx.ckpt',num_classes=2)
model_all = Pretrain_model.load_from_checkpoint(model_path + 'model_all.ckpt')

result_all = test_cnn(model_all,test_dl,times=1)
result_hw = test_cnn(model_hw,test_dl,times=1)
result_wx = test_cnn(model_wx,test_dl,times=1)


for i in range(len(result_all)):
    result_all[i,2] = result_hw[i,1]
    result_all[i,3] = result_wx[i,1]


####### writing outputs ##############
image_names = os.listdir(test_path)
#labels_dict = {'dc':0,'hntc':1,'hwylc':2,'wxpc':3,'yyc':4}
result = torch.argmax(result_all,1)
labels_dict = {0:'dc',1:'hntc',2:'hwylc',3:'wxpc',4:'yyc'}
submit_result = []
for i,name in enumerate( image_names):
    one = {}
    one['image_name'] = name
    one['category'] = labels_dict[int(result[i])]
    submit_result.append(one)
with open('training_results.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(submit_result))


