import torch
from model import model_mobilefaceNet
from model.model_integrated import Adv_Model
import torchvision.transforms as T
from PIL import Image
import numpy as np

def read_img(imgPath): 
    data = torch.randn(1, 3, 112, 112)
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(imgPath).convert("RGB")
    data[0, :, :, :] = transform(img)
    return data

def network(eval_model, device):
    net = model_mobilefaceNet.MobileFaceNet([112, 112], 512, output_name = 'GDC').to(device)
    adv_net = Adv_Model(net, type='Test').to(device)
    net_dict = adv_net.state_dict()     
    data_dict = {key.replace('module.', ''): value for key, value in torch.load(eval_model, map_location=device).items()}
    net_dict.update(data_dict)
    adv_net.load_state_dict(net_dict)
    adv_net.eval()
    return adv_net

if __name__ == "__main__":
    imgpath = './demo_imgs/Afr1.jpg'                         # [Afr1.jpg, Afr2.jpg, Asi1.jpg, Asi2.jpg, Ind1.jpg, Ind2.jpg]
    device = 'cpu'                                           # 'cpu' or 'cuda:x'
    eval_model = './checkpoints/EQBM-S-African.pth'          # checkpoint
    net = network(eval_model, device)
    input_data = read_img(imgpath)
    anchor = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
    pred_dist = net(input_data, train=False).data.cpu().numpy().squeeze()
    pred_score = np.sum(anchor * pred_dist)
    print(f"Quality score = {pred_score}")