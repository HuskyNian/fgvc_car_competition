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

#### random seeds
torch.backends.cudnn.deterministic = True
seed = 42
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed(42)
np.random.seed(seed)
def seed_worker(worker_id):
    global seed
    numpy.random.seed(seed)
    random.seed(wseed)
g = torch.Generator()
g.manual_seed(seed)

train_path = '/car_data/images/'
model_path = '/car_data/train_models/'
labels_dict = {'dc':0,'hntc':1,'hwylc':2,'wxpc':3,'yyc':4}
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
labels = pd.read_csv('/car_data/test_labels.csv')
labels['label'] = labels['label'].apply(lambda x:labels_dict[x])
print(labels.head())
print(labels.label.unique())

# data
X = labels.filename.to_list()
y = labels.label.to_list()

################### data argumentation#################
transform_train = A.Compose([
    A.Resize(256, 256), # Resize image
    A.OneOf([A.RandomCrop(224,224),
             A.Resize(224,224)
    ],p=1),
    #A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.5),
    A.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.5),
    A.HorizontalFlip(p=0.5), # Horizontal Symmetric Conversion
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=25, p=0.7),
    A.OneOf([A.Emboss(p=1),
             A.Sharpen(p=1),
             A.Blur(p=1)], p=0.5),
    #A.PiecewiseAffine(p=0.5), # Affine Transformation 
    A.Normalize(), # Normalize Transformation 
    ToTensorV2() # Convert to Tensor
])
transform_test = A.Compose([
    A.Resize(224, 224), # Resize image
    #A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.5),
    A.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.5),
    A.HorizontalFlip(p=0.5), # Horizontal Symmetric Conversion
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=25, p=0.7),
    A.OneOf([A.Emboss(p=1),
             A.Sharpen(p=1),
             A.Blur(p=1)], p=0.5),
    #A.PiecewiseAffine(p=0.5), # Affine Transformation 
    A.Normalize(), # Normalize Transformation 
    ToTensorV2() # Convert to Tensor
])
resize = transforms.Resize((224,224))
totensor = transforms.ToTensor()
toPIL = transforms.ToPILImage()
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class Augmented_dataset(Dataset):
    def __init__(self, imgs,labels,path,test=False):
        self.imgs = imgs
        self.labels = labels
        self.length = len(labels)
        self.path = path
        self.test = test
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.labels[idx]
        image = cv2.imread(self.path +image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.test:
            image = transform_test(image=image)['image']
        else:
            image = transform_train(image=image)['image']
        return image,label
    
class Original_dataset(Dataset):
    def __init__(self, imgs,labels,path):
        self.imgs = imgs
        self.labels = labels
        self.length = len(labels)
        self.path = path
        
    def __len__(self):
        # RETURN SIZE OF DATASET
        return self.length

    def __getitem__(self, idx):
        # RETURN IMAGE AT GIVEN idx
        image = self.imgs[idx]
        label = self.labels[idx]
        image = Image.open(self.path +image)
        
        # augmentation
        image = resize(image)
        image = totensor(image)
        image = normalize(image)
        return image,label

train_ds = Augmented_dataset(X,y,train_path)
val_ds = Original_dataset(X,y,train_path)
batch_size = 2
train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=0,
worker_init_fn=seed_worker,generator=g)
val_dl = DataLoader(val_ds,batch_size=batch_size,shuffle=False)

################## model configuration ###############
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



################# Training process#####################
def train_cnn(model,train_dl,val_dl,ep=15,ckpt=None):
    if ckpt:
        trainer = pl.Trainer(max_epochs = ep,gpus = 1,precision = 16,auto_lr_find=True,
                callbacks=[ckpt])
    else:
        trainer = pl.Trainer(max_epochs = ep,gpus = 1,precision = 16,auto_lr_find=True)
                #callbacks=[early_stop_callback])
    trainer.fit(model,train_dl,val_dl)
    return trainer

class LabelSmoothingLoss_nvidia(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingLoss_nvidia, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def test_cnn(model,dl,val=True,times=5):
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

def make_dls(target_class,X,y):
    y_hw = np.array(y.copy(),dtype=np.int64)
    y_hw[np.where(y_hw!=target_class)] = 0
    y_hw[np.where(y_hw==target_class)] = 1

    train_ds_hw = Augmented_dataset(X,y_hw,train_path)
    val_ds_hw = Original_dataset(X,y_hw,train_path)

    train_dl_hw = DataLoader(train_ds_hw,batch_size=batch_size,shuffle=True,num_workers=0)
    val_dl_hw =   DataLoader(val_ds_hw,batch_size = batch_size,shuffle = False)
    return train_dl_hw,val_dl_hw,y_hw
######### training for wx class #############
#labels_dict = {'dc':0,'hntc':1,'hwylc':2,'wxpc':3,'yyc':4}
model_wx = Pretrain_model(lr = 0.0001,use_timm=True,loss_fun=LabelSmoothingLoss_nvidia(0.15),num_classes=2)
train_dl_wx,val_dl_wx,y_wx = make_dls(3,X,y)
trainer = train_cnn(model_wx,train_dl_wx,val_dl_wx,ep=13)

#save model
trainer.save_checkpoint(model_path + "model_wx.ckpt")
result_wx = test_cnn(model_wx,val_dl_wx,times=1)
print(f1_score(y_wx,torch.argmax(result_wx,1),average ='micro'))
######### training for hw class ##########
model_hw = Pretrain_model(lr = 0.0001,use_timm=True,loss_fun=LabelSmoothingLoss_nvidia(0.15),num_classes=2)
train_dl_hw,val_dl_hw,y_hw = make_dls(2,X,y)
trainer = train_cnn(model_hw,train_dl_hw,val_dl_hw,ep=4)
trainer.save_checkpoint(model_path + "model_hw.ckpt")

result_hw = test_cnn(model_hw,val_dl_hw,times=1)
print(f1_score(y_hw,torch.argmax(result_hw,1),average ='micro'))

########### training for all classes ##########
model_all = Pretrain_model(lr = 0.0001,use_timm=False,loss_fun=LabelSmoothingLoss_nvidia(0.15),num_classes=5)
trainer = train_cnn(model_all,train_dl,val_dl,ep=10)
trainer.save_checkpoint(model_path + "model_all.ckpt")

result_all = test_cnn(model_all,val_dl,times=1)
print(f1_score(y,torch.argmax(result_all,1),average ='micro'))


######## ensembling all results ##############
for i in range(len(result_all)):
    result_all[i,2] = result_hw[i,1]
    result_all[i,3] = result_wx[i,1]
print(f1_score(y,torch.argmax(result_all,1),average ='micro'))






