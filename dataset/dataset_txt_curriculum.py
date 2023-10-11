import random
import torch
from torch.utils.data import DataLoader
from torch.utils import data
from PIL import Image
import numpy as np


class Dataset(data.Dataset):
    def __init__(self, conf, curr_rec, curriculum_type='easy'):
        super().__init__()
        self.curriculum_type = curriculum_type
        self.transform = conf.transform
        self.batch_size = conf.batch_size

        self.imgPath_source = []
        self.imgPath_target = []
        self.inter_uncertainty_source = []
        self.inter_uncertainty_target = []
        self.quality_source = []
        self.quality_target = []

        imgPath_source_GT, quality_source_GT = [], []
        with open(conf.source_list, 'r') as f: txtContent = f.readlines()
        for line in txtContent:
            value = line.split()
            imgPath_source_GT.append(value[0])
            quality_source_GT.append(float(value[1]))
        tmp_GT = np.asarray(quality_source_GT)
        tmp_GT = (tmp_GT - np.min(tmp_GT)) / (np.max(tmp_GT) - np.min(tmp_GT))

        self.source_GT = {}
        for i in range(len(imgPath_source_GT)): self.source_GT[imgPath_source_GT[i]] = tmp_GT[i]
        
        for value in curr_rec:
            if value[0] == 'Source':
                self.imgPath_source.append(value[1])
                self.inter_uncertainty_source.append(float(value[2]))
                self.quality_source.append(float(value[3]))
            else: 
                self.imgPath_target.append(value[1])
                self.inter_uncertainty_target.append(float(value[2]))
                self.quality_target.append(float(value[3]))
        
        self.imgPath_source = np.asarray(self.imgPath_source)
        self.inter_uncertainty_source = np.asarray(self.inter_uncertainty_source)
        self.quality_source = np.asarray(self.quality_source)
        self.imgPath_target = np.asarray(self.imgPath_target)
        self.inter_uncertainty_target = np.asarray(self.inter_uncertainty_target)
        self.quality_target = np.asarray(self.quality_target)

        inter_threshold_src = np.median(self.inter_uncertainty_source)
        inter_threshold_tag = np.median(self.inter_uncertainty_target)

        if curriculum_type == 'easy':
            sample_src_idx = np.argwhere(self.inter_uncertainty_source<=inter_threshold_src)
            sample_tag_idx = np.argwhere(self.inter_uncertainty_target<=inter_threshold_tag)
            self.quality_th = 2
        elif curriculum_type == 'hard':
            sample_src_idx = np.argwhere(self.inter_uncertainty_source>inter_threshold_src)
            sample_tag_idx = np.argwhere(self.inter_uncertainty_target>inter_threshold_tag)
            self.quality_th = 1

        self.imgPath_source = self.imgPath_source[sample_src_idx].squeeze()
        self.inter_uncertainty_source = self.inter_uncertainty_source[sample_src_idx].squeeze()
        self.quality_source = self.quality_source[sample_src_idx].squeeze()

        self.imgPath_target = self.imgPath_target[sample_tag_idx].squeeze()
        self.inter_uncertainty_target = self.inter_uncertainty_target[sample_tag_idx].squeeze()
        self.quality_target = self.quality_target[sample_tag_idx].squeeze()

        self.quality_norm_source = (self.quality_source - np.mean(self.quality_source)) / np.std(self.quality_source)
        self.quality_norm_target = (self.quality_target - np.mean(self.quality_target)) / np.std(self.quality_target)
        


    def __getitem__(self, index):
        imgPath_target1  = self.imgPath_target[index]
        quality_target1 = self.quality_target[index]
        quality_norm_target1 = self.quality_norm_target[index]

        selected_idx = np.argwhere(abs(self.quality_norm_target - quality_norm_target1)>self.quality_th).squeeze()
        rand_one_idx = np.random.randint(0 ,len(selected_idx))
        selected_idx = selected_idx[rand_one_idx]
        imgPath_target2  = self.imgPath_target[selected_idx]
        quality_target2 = self.quality_target[selected_idx]
        quality_norm_target2 = self.quality_norm_target[selected_idx]

        selected_idx = list(range(0 ,len(self.imgPath_source)))
        rand_one_idx = np.random.randint(0 ,len(self.imgPath_source))
        selected_idx_src = selected_idx[rand_one_idx]
        imgPath_source1  = self.imgPath_source[selected_idx_src]
        quality_source1 = self.quality_source[selected_idx_src]
        quality_norm_source1 = self.quality_norm_source[selected_idx_src]
        selected_idx = np.argwhere(abs(self.quality_norm_source - quality_norm_source1)>self.quality_th).squeeze()
        rand_one_idx = np.random.randint(0 ,len(selected_idx))
        selected_idx = selected_idx[rand_one_idx]
        imgPath_source2  = self.imgPath_source[selected_idx]
        quality_source2 = self.quality_source[selected_idx]
        quality_norm_source2 = self.quality_norm_source[selected_idx]

        img_source1 = Image.open(imgPath_source1).convert("RGB")
        img_source2 = Image.open(imgPath_source2).convert("RGB")
        img_target1 = Image.open(imgPath_target1).convert("RGB")
        img_target2 = Image.open(imgPath_target2).convert("RGB")

        img_source1 = self.transform(img_source1)
        img_source2 = self.transform(img_source2)
        img_target1 = self.transform(img_target1)
        img_target2 = self.transform(img_target2)
        quality_source1 = self.source_GT[imgPath_source1]
        quality_source2 = self.source_GT[imgPath_source2]

        return img_source1, img_source2, img_target1, img_target2, quality_source1, quality_source2, quality_target1, quality_target2, (imgPath_source1, imgPath_source2, imgPath_target1, imgPath_target2)


    def __len__(self):
        return(len(self.imgPath_target))
                
def load_data(conf, curr_rec, curriculum_type='easy'):
    dataset = Dataset(conf, curr_rec, curriculum_type=curriculum_type)

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
    loader = (loader1, loader2)

    return loader
