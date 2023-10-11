import torch
import torchvision.transforms as T

class Config(object):
# Training dataset 
    source_list = '/Finch/Dataset/BT_Source_SDD_pseudo-labels.txt'               
    target_list = '/Finch/Dataset/BT_Target.txt'
    source_label = 'Caucasian'
    target_label = 'African'     #[African, Asian, Indian]

# Model settings
    pretrain_model = '/Finch/Dataset/FR_models/MobileFaceNet_MS1M.pth'
    checkpoints = f"/Finch/EQBM/checkpoints/{target_label}"

# Data preprocess
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_target = T.Compose([
        T.RandomHorizontalFlip(),
        # T.Resize([112,112]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

# Training settings
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = 1024
    multi_GPUs = [0]
    pin_memory = True
    num_workers = 8
    batch_size = 128 
    epoch = 18
    lr = 0.0001 
    stepLR = [10, 15]
    weight_decay = 0.0005
    saveModel_epoch = 1

config = Config()
