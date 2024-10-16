import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import  Dataset
from torch import from_numpy as fn
from sklearn.decomposition import PCA
from config import load_args
import torch.nn as nn
from augment import *
from utils import *

args = load_args() 

def pca_whitening(image, number_of_pc):

    shape = image.shape 
    image = np.reshape(image, [shape[0]*shape[1], shape[2]])
    number_of_rows = shape[0]
    number_of_columns = shape[1]
    pca = PCA(n_components = number_of_pc)
    image = pca.fit_transform(image)
    pc_images = np.zeros(shape=(number_of_rows, number_of_columns, number_of_pc),dtype=np.float32)
    for i in range(number_of_pc):
        pc_images[:, :, i] = np.reshape(image[:, i], (number_of_rows, number_of_columns))
    
    return pc_images  

def load_data(datasets):
        
    if datasets == 'PaviaU':
        data_path = 'datasets\PaviaU\PaviaU.mat'
        gt_path = 'datasets\PaviaU\PaviaU_gt.mat'
        
    data = {k: v for k, v in sio.loadmat(data_path).items()
                     if isinstance(v, np.ndarray)}
    gt = {k: v for k, v in sio.loadmat(gt_path).items()
                if isinstance(v, np.ndarray) and 'map' not in k} 
    
    data, gt = list(data.values())[0], list(gt.values())[0]

    nbands = data.shape[2]
    nclass = np.max(gt)
    
    return data_path, gt_path, nbands, nclass

def get_datasets(datasets,datatype,seed,**kwargs):
    data_path, gt_path,_,_ = load_data(datasets)
    return HSIdataset(data_path,gt_path,args.windowsize,args.ratio_train,datatype,seed,**kwargs)

class HSIdataset(Dataset):
    
    def __init__(self,
                 data_path,
                 gt_path,
                 windowsize,
                 ratio_pertrain,
                 datatype,
                 seed,
                 augment=True,
                 ):
        self.augment = augment
        self.data_path = data_path
        self.gt_path = gt_path
             
        super(HSIdataset,self).__init__()
        self.windowsize = windowsize
        halfsize = windowsize // 2
        self.halfsize = halfsize
        self.seed = seed
        data = {k: v for k, v in sio.loadmat(data_path).items()
                     if isinstance(v, np.ndarray)}
        gt = {k: v for k, v in sio.loadmat(gt_path).items()
                   if isinstance(v, np.ndarray) and 'map' not in k}
        
        data, gt = list(data.values())[0], list(gt.values())[0]

        nclass = np.max(gt)

        self.bands = data.shape[2]
       
        data = np.pad(data, ((halfsize, halfsize), (halfsize, halfsize), (0, 0)), 'edge')
        gt = np.pad(gt, ((halfsize, halfsize), (halfsize, halfsize)), 'constant',constant_values=0)

        n = np.zeros(nclass,dtype=np.int64)
        for i in range(nclass):
            coord_x,coord_y = np.where(gt == i + 1)
            n[i] = len(coord_x)
        
        data = 1 * ((data - np.min(data)) / (np.max(data) - np.min(data)) )
        self.image = data
        self.data = data.transpose([2, 0, 1])
        pca_data = pca_whitening(data,args.c_pca)
        self.pca_data = pca_data.transpose([2, 0, 1])
        self.gt_pad = gt
        self.gt = gt - 1
        ntrain_perclass = np.ones(nclass,dtype=np.int64)
        self.datatype = datatype
        coords_x = np.array([],dtype=np.int64)
        coords_y = np.array([],dtype=np.int64)
        len_test = np.zeros(nclass,dtype=np.int64)
        for i in range(nclass):
            ntrain_perclass[i] = max(n[i] * ratio_pertrain,15)
            RandPerm_per = torch.randperm(
                n[i], dtype=torch.int,
                generator=torch.Generator().manual_seed(seed),
            ).numpy().squeeze()
            coord_x,coord_y = np.where(gt == i + 1)
            if datatype == 'train':
                selected_x = coord_x[RandPerm_per[:ntrain_perclass[i]]]
                selected_y = coord_y[RandPerm_per[:ntrain_perclass[i]]]
            elif datatype =='test':
                selected_x = coord_x[RandPerm_per[ntrain_perclass[i]:]]
                selected_y = coord_y[RandPerm_per[ntrain_perclass[i]:]]
            len_test[i] = len(selected_x)
            
            coords_x = np.concatenate([coords_x, selected_x])
            coords_y = np.concatenate([coords_y, selected_y])
        self.len_test = len_test
        self.coords_x = coords_x
        self.coords_y = coords_y
        self.pooling = nn.MaxPool2d(3)
        self.labels = self.gt.flatten()
                
    def __getitem__(self,item):
        coords_x, coords_y = self.coords_x[item], self.coords_y[item]
        halfsize = self.halfsize
        halfsize2 = 1

        spa_data = fn(self.pca_data[:,
                                coords_x - halfsize : coords_x + halfsize + 1, 
                                coords_y - halfsize : coords_y + halfsize + 1].astype(np.float32))
        
        spe_data = fn(self.data[:,
                                coords_x - halfsize2 : coords_x + halfsize2 + 1, 
                                coords_y - halfsize2 : coords_y + halfsize2 + 1].astype(np.float32))

        spe_data = self.pooling(spe_data)
        spe_data = spe_data.squeeze(-1)
        spe_data.transpose_(0,1)
        
        if self.augment == True and self.datatype == 'train':
            transfor_data = CenterResizeCrop(scale_begin = 17, windowsize = self.windowsize)
            spa_data = fn(np.asarray(transfor_data(spa_data.numpy())))
        label = self.gt[coords_x, coords_y].astype(np.int64)
        label = fn(np.array(label))
        
        return spa_data, spe_data, label
        
    def __len__(self):
        return len(self.coords_x)
