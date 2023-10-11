import torch
from torch.utils.data import DataLoader
from torch.utils import data
from PIL import Image
import numpy as np
import random

class Dataset(data.Dataset):
    def __init__(self, conf, train, label=True, target_domain=False):
        super().__init__()
        self.train = train
        if self.train:
            self.img_list = conf.source_list
            self.img_list_t = conf.target_list
        else:
            self.img_list = conf.test_list
        self.transform = conf.transform
        
        self.batch_size = conf.batch_size
        self.label = label
        self.target_domain = target_domain

        with open(self.img_list, 'r') as f:
            self.imgPath = []
            self.imgPath_target = []
            self.target = []
            for index, value in enumerate(f):

                value = value.split()
                if self.label:
                    if value and len(value) < 2:
                        print(f"ERROR, {value}({index}-th) is missing, please check it")
                    else:
                        self.imgPath.append(value[0])
                        self.target.append(float(value[1]))                    
                else:
                    self.imgPath.append(value[0])
                    self.target.append(float(0))

            self.target = np.asarray(self.target)
            self.target = (self.target - np.min(self.target)) / (np.max(self.target) - np.min(self.target))

            if self.target_domain:
                self.transform_target = conf.transform_target
                with open(self.img_list_t, 'r') as f: 
                    target_imgs = []
                    for index, value in enumerate(f):
                        value = value.split()
                        if conf.target_label in value[0]:
                            target_imgs.append(value[0])
                        else:
                            continue
                rnd_idx = np.random.randint(0, len(target_imgs), len(self.imgPath))
                self.imgPath_target = [target_imgs[i] for i in rnd_idx]           

    def __getitem__(self, index):
        imgPath = self.imgPath[index]
        img = Image.open(imgPath).convert("RGB")
        assert img.size[0] == 112
        source_img = self.transform(img)
        source_labels = self.target[index]

        imgPath_target = self.imgPath_target[index]
        img_target = Image.open(imgPath_target).convert("RGB")
        assert img_target.size[0] == 112
        target_img = self.transform_target(img_target)
        return imgPath, source_img, source_labels, imgPath_target, target_img

    def __len__(self):
        return(len(self.imgPath))
                
def load_data(conf,train,label,target):
    dataset = Dataset(conf,train,label,target)
    if train:
        loader_all = DataLoader(dataset, 
                        batch_size=conf.batch_size, 
                        shuffle=True, 
                        pin_memory=conf.pin_memory, 
                        num_workers=conf.num_workers)

        train_ratio = 0.9
        train_size = int(train_ratio * len(dataset))
        indices = random.sample(range(len(dataset)), len(dataset))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        loader1 = DataLoader(train_dataset, 
                        batch_size=conf.batch_size, 
                        shuffle=True, 
                        pin_memory=conf.pin_memory, 
                        num_workers=conf.num_workers)
        loader2 = DataLoader(val_dataset, 
                        batch_size=conf.batch_size, 
                        shuffle=True, 
                        pin_memory=conf.pin_memory, 
                        num_workers=conf.num_workers)
        loader = (loader1, loader2, loader_all)

    else:
        loader = DataLoader(dataset, 
                        batch_size=conf.batch_size, 
                        shuffle=False, 
                        pin_memory=conf.pin_memory, 
                        num_workers=conf.num_workers)
    return loader

