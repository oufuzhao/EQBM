import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import model_mobilefaceNet

from dataset.dataset_txt import load_data as load_data_txt0
from dataset.dataset_txt_curriculum import load_data as load_data_txt
from config_train import config as conf
import numpy as np
from utilities import AverageMeter
from model.model_integrated import Adv_Model


def dataSet0(conf):
    trainloader = load_data_txt0(conf, train=True, label=True, target=True)
    return trainloader

def dataSet(conf, curr_rec):
    trainloader_easy = load_data_txt(conf, curr_rec, curriculum_type='easy')
    trainloader_hard = load_data_txt(conf, curr_rec, curriculum_type='hard')
    return (trainloader_easy, trainloader_hard)

def backboneSet(conf, type=None):
    device = conf.device
    multi_GPUs = conf.multi_GPUs
    net_type = type
    net = model_mobilefaceNet.MobileFaceNet([112, 112], 512, output_name = 'GDC').to(device)
    print(net)
    if conf.pretrain_model != None:
        net_dict = net.state_dict()
        pretrained_dict = torch.load(conf.pretrain_model, map_location=device)
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        same_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
        diff_dict = {k: v for k, v in net_dict.items() if k not in pretrained_dict}
        net_dict.update(same_dict)
        net.load_state_dict(net_dict)
    adv_net = Adv_Model(net, type = net_type).to(device)
    if device != 'cpu' and len(multi_GPUs) > 1:    # mulit-GPUs processing
        adv_net = nn.DataParallel(net, device_ids=multi_GPUs)    
    return adv_net


def trainSet(conf, net, l_lr=False):
    # Loss
    criterion1 = Dist_distance
    criterion2 = torch.nn.NLLLoss()
    criterion = (criterion1, criterion2)
    # Optimizer
    lr = conf.lr
    if l_lr: lr *= 0.01
    optimizer = optim.Adam(net.parameters(),
                           lr=lr,
                           weight_decay=conf.weight_decay)
    scheduler_gamma = 0.1
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=conf.stepLR, gamma=scheduler_gamma)
    return criterion, optimizer, scheduler


def Dist_distance(p, q, r=2):
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    cdf_p = torch.cumsum(p, dim=1)
    cdf_q = torch.cumsum(q, dim=1)
    cdf_diff = torch.abs(cdf_p - cdf_q)
    cdf_diff = torch.mean(cdf_diff ** r, dim=1)
    single_dist = cdf_diff ** (1. / r)
    return torch.mean(single_dist)

def score_to_dist(x):
    x = torch.reshape(x, [-1,1])
    anchor_bins = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]).to(conf.device)
    anchor_bins = anchor_bins.repeat(x.size(0), 1).to(conf.device)
    beta = torch.tensor(-64).to(conf.device)
    dist_anchors = torch.exp(beta * torch.square(x - anchor_bins)).to(conf.device)
    norm_dist_anchors = dist_anchors / torch.reshape(torch.sum(dist_anchors, dim=1), [-1, 1]).to(conf.device)
    return norm_dist_anchors

def dist_to_score(x):
    anchor_bins = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]).to(conf.device)
    anchor_bins = anchor_bins.repeat(x.size(0), 1).to(conf.device)
    norm_scores = x * anchor_bins
    one_scores = torch.sum(norm_scores, dim=1).to(conf.device)
    return one_scores

def train_curr_measurers(conf, trainloader, net, epoch):
    net.train()
    rec_losses_1 = AverageMeter()
    rec_losses_2 = AverageMeter()
    rec_losses_3 = AverageMeter()
    val_loss_list = [10.0]
    dec_num = 3
    for e in range(int(epoch)):
        itersNum = 1
        for _, source_data, source_labels, _, target_data in trainloader[0]:
            source_data = source_data.to(conf.device)
            target_data = target_data.to(conf.device)
            source_labels = source_labels.to(conf.device).to(torch.float32) #/ 100
            dist_source_labels = score_to_dist(source_labels)
            source_domain_labels = torch.zeros(source_data.size(0)).long().to(conf.device)
            target_domain_labels = torch.ones(source_data.size(0)).long().to(conf.device)
            source_class_output, source_domain_output, target_domain_output = net(input_data1=source_data, input_data2=target_data, alpha=None, train=True)
            loss1 = criterion[0](source_class_output, dist_source_labels)
            loss_source_domain = criterion[1](source_domain_output, source_domain_labels)
            loss_target_domain = criterion[1](target_domain_output, target_domain_labels)
            loss2 = loss_source_domain + loss_target_domain
            total_loss = loss1 + loss2
            
            rec_losses_1.update(loss1.data.item(), source_data.size(0))
            rec_losses_2.update(loss2.data.item(), source_data.size(0))
            rec_losses_3.update(total_loss.data.item(), source_data.size(0))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if itersNum % (int(len(trainloader[0])/20)+2) == 0:
                print(f"Epo/Its = {e+1}/{itersNum} | LR = {optimizer.param_groups[0]['lr']} | Loss1 = {rec_losses_1.avg:.5f} | Loss2 = {rec_losses_2.avg:.5f} | All_Loss = {rec_losses_3.avg:.5f}")
            itersNum += 1

        rec_losses_val = AverageMeter()
        for _, source_data, source_labels, _, target_data in trainloader[1]:
            net.eval()
            with torch.no_grad():
                source_data = source_data.to(conf.device)
                target_data = target_data.to(conf.device)
                source_labels = source_labels.to(conf.device).to(torch.float32)
                dist_source_labels = score_to_dist(source_labels)
                source_domain_labels = torch.zeros(source_data.size(0)).long().to(conf.device)
                target_domain_labels = torch.ones(source_data.size(0)).long().to(conf.device)
                source_class_output, source_domain_output, target_domain_output = net(input_data1=source_data, input_data2=target_data, alpha=None, train=True)
                loss1 = criterion[0](source_class_output, dist_source_labels)
                loss_source_domain = criterion[1](source_domain_output, source_domain_labels)
                loss_target_domain = criterion[1](target_domain_output, target_domain_labels)
                loss2 = loss_source_domain + loss_target_domain
                total_val_loss = loss1 + loss2
                rec_losses_val.update(total_val_loss.data.item(), source_data.size(0))
        net.train()

        val_loss_list.append(rec_losses_val.avg)
        if rec_losses_val.avg >= val_loss_list[-2]: 
            net.load_state_dict(save_weights)
            for param_group in optimizer.param_groups: 
                dec_num -= 1
                param_group['lr'] = optimizer.param_groups[0]['lr'] * 0.1
            if dec_num == 0: return net
            else: continue
        save_weights = net.state_dict()
        scheduler.step()
    return net


def curr_design(trainloader, net):
    net.eval()
    with torch.no_grad(): 
        param = {}
        for name, parameters in net.named_parameters(): param[name] = parameters
        learned_center_weight = param['uncertainty_domain_classifier_2.0.weight']
        source_center_feats = learned_center_weight[0, :].to(conf.device)
        domain_center_feats = learned_center_weight[1, :].to(conf.device)
        dict_src_l2, dict_tag_l2, qua_src_score, qua_tag_score = {}, {}, {}, {}

        for imgPath, source_data, source_labels, imgPath_target, target_data in trainloader[2]:
            source_data = source_data.to(conf.device)
            target_data = target_data.to(conf.device)
            source_labels = source_labels.to(conf.device).to(torch.float32).cpu().detach().numpy()
            _, src_inter_feats = net(input_data1=source_data, train=False)
            tag_output, tag_inter_feats = net(input_data1=target_data, train=False)
            tag_output = dist_to_score(tag_output).cpu().detach().numpy()
            # calculate l2 distance
            source_center_feats_anchor = source_center_feats.repeat(source_data.size(0), 1).to(conf.device)
            target_center_feats_anchor = domain_center_feats.repeat(source_data.size(0), 1).to(conf.device)
            src_inter_l2 = torch.sqrt(torch.sum((target_center_feats_anchor - src_inter_feats)**2, dim=1)).cpu().detach().numpy()
            tag_inter_l2 = torch.sqrt(torch.sum((source_center_feats_anchor - tag_inter_feats)**2, dim=1)).cpu().detach().numpy()       

            for i in range(len(imgPath)):
                dict_src_l2[imgPath[i]] = src_inter_l2[i]
                dict_tag_l2[imgPath_target[i]] = tag_inter_l2[i]
                qua_src_score[imgPath[i]] = source_labels[i]
                qua_tag_score[imgPath_target[i]] = tag_output[i]
        src_rec = []
        tag_rec = []        
        for i, v in enumerate(list(dict_src_l2.keys())):
            src_rec.append(['Source', str(v), str(dict_src_l2[v]), str(qua_src_score[v])])
        src_rec = np.asarray(src_rec)
        for i, v in enumerate(list(dict_tag_l2.keys())):
            tag_rec.append(['Target', str(v), str(dict_tag_l2[v]), str(qua_tag_score[v])])
        tag_rec = np.asarray(tag_rec)
        rec = np.concatenate((src_rec, tag_rec), axis=0)
        return rec


def train_da(conf, trainloader, net, epoch):
    print('='*20 + 'TRAINING LOG' + '='*20)
    net.train()
    rec_losses_1 = AverageMeter()
    rec_losses_2 = AverageMeter()
    rec_losses_3 = AverageMeter()
    rec_losses_4 = AverageMeter()
    val_loss_list = [10.0]
    dec_num = 3

    for e in range(int(epoch)):
        itersNum = 1
        for img_source1, img_source2, img_target1, img_target2, quality_source1, quality_source2, quality_target1, quality_target2, _ in trainloader[0]:
            p = float(itersNum-1 + 4 * len(trainloader[0])) / 4 / len(trainloader[0])
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            img_source1 = img_source1.to(conf.device)
            img_source2 = img_source2.to(conf.device)
            img_target1 = img_target1.to(conf.device)
            img_target2 = img_target2.to(conf.device)
            quality_source1 = quality_source1.to(conf.device).to(torch.float32)
            quality_source2 = quality_source2.to(conf.device).to(torch.float32)
            quality_target1 = quality_target1.to(conf.device).to(torch.float32)
            quality_target2 = quality_target2.to(conf.device).to(torch.float32)
            dist_source_labels1 = score_to_dist(quality_source1)

            source_class_output1, _, rank_classifier_output, source_domain_output1, source_domain_output2 = net(input_data1=img_source1, input_data2=img_source2, alpha=alpha)
            _, _, _, target_domain_output1, target_domain_output2 = net(input_data1=img_target1, input_data2=img_target2, alpha=alpha)

            EMD_loss = criterion[0](source_class_output1, dist_source_labels1) 
            rank_orders = torch.gt(quality_source1, quality_source2).long().to(conf.device)
            loss_ranking = criterion[1](rank_classifier_output, rank_orders)

            f_s = torch.concat([source_domain_output1, source_domain_output2], dim=0)
            f_t = torch.concat([target_domain_output1, target_domain_output2], dim=0)
            source_domain_labels = torch.zeros(2 * img_source1.size(0)).long().to(conf.device)
            target_domain_labels = torch.ones(2 * img_source1.size(0)).long().to(conf.device)
            loss_source_domain = 0.8 * criterion[1](f_s, source_domain_labels)
            loss_target_domain = 0.8 * criterion[1](f_t, target_domain_labels)

            loss_domain = loss_source_domain + loss_target_domain
            total_loss = EMD_loss + loss_ranking + loss_domain

            rec_losses_1.update(EMD_loss.data.item(), img_source1.size(0))
            rec_losses_2.update(loss_ranking.data.item(), img_source1.size(0))
            rec_losses_3.update(loss_domain.data.item(), img_source1.size(0))
            rec_losses_4.update(total_loss.data.item(), img_source1.size(0))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if itersNum % (int(len(trainloader[0])/20)+2) == 0:
                print(f"Epo/Its = {e+1}/{itersNum} | LR = {optimizer.param_groups[0]['lr']} | EMD_Loss = {rec_losses_1.avg:.5f} | Rank_Loss = {rec_losses_2.avg:.5f} | Adv_Loss = {rec_losses_3.avg:.5f} | All_Loss = {rec_losses_4.avg:.5f}")
            itersNum += 1

        rec_losses_val = AverageMeter()
        net.eval()
        for img_source1, img_source2, img_target1, img_target2, quality_source1, quality_source2, quality_target1, quality_target2, _ in trainloader[1]:
            with torch.no_grad():
                img_source1 = img_source1.to(conf.device)
                img_source2 = img_source2.to(conf.device)
                img_target1 = img_target1.to(conf.device)
                img_target2 = img_target2.to(conf.device)
                quality_source1 = quality_source1.to(conf.device).to(torch.float32)
                quality_source2 = quality_source2.to(conf.device).to(torch.float32)
                quality_target1 = quality_target1.to(conf.device).to(torch.float32)
                quality_target2 = quality_target2.to(conf.device).to(torch.float32)
                dist_source_labels1 = score_to_dist(quality_source1)
                source_class_output1, _, rank_classifier_output, source_domain_output1, source_domain_output2 = net(input_data1=img_source1, input_data2=img_source2, alpha=alpha)
                _, _, _, target_domain_output1, target_domain_output2 = net(input_data1=img_target1, input_data2=img_target2, alpha=alpha)
                EMD_loss = criterion[0](source_class_output1, dist_source_labels1) 
                rank_orders = torch.gt(quality_source1, quality_source2).long().to(conf.device)
                loss_ranking = criterion[1](rank_classifier_output, rank_orders)
                f_s = torch.concat([source_domain_output1, source_domain_output2], dim=0)
                f_t = torch.concat([target_domain_output1, target_domain_output2], dim=0)
                source_domain_labels = torch.zeros(2 * img_source1.size(0)).long().to(conf.device)
                target_domain_labels = torch.ones(2 * img_source1.size(0)).long().to(conf.device)
                loss_source_domain = 0.8 * criterion[1](f_s, source_domain_labels)
                loss_target_domain = 0.8 * criterion[1](f_t, target_domain_labels)
                loss_domain = loss_source_domain + loss_target_domain
                total_val_loss = EMD_loss + loss_ranking + loss_domain
                rec_losses_val.update(total_val_loss.data.item(), img_source1.size(0))
        net.train()
        val_loss_list.append(rec_losses_val.avg)
        if rec_losses_val.avg >= val_loss_list[-2]: 
            net.load_state_dict(save_weights)
            for param_group in optimizer.param_groups: 
                dec_num -= 1
                param_group['lr'] = optimizer.param_groups[0]['lr'] * 0.1
            if dec_num == 0: return net
            else: continue
        print(f"Epo/Its = {e+1}/{itersNum} | LR = {optimizer.param_groups[0]['lr']} | EMD_Loss = {rec_losses_1.avg:.5f} | Rank_Loss = {rec_losses_2.avg:.5f} | Adv_Loss = {rec_losses_3.avg:.5f} | All_Loss = {rec_losses_4.avg:.5f}")
        save_weights = net.state_dict()
        if (e+1) % conf.saveModel_epoch == 0:   # save model
            os.makedirs(conf.checkpoints, exist_ok=True)
            savePath = os.path.join(conf.checkpoints, f"EQBM.pth")
            if len(conf.multi_GPUs)>1:
                torch.save(net.module.state_dict(), savePath)
            else:
                torch.save(net.state_dict(), savePath)
            print(f"SAVE MODEL: {savePath}")
        scheduler.step()
    return net


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    import random
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(conf.seed)
    
    trainloader = dataSet0(conf)
    curr_net = backboneSet(conf, type='Curr_Measurers')
    criterion, optimizer, scheduler = trainSet(conf, curr_net)
    net_cum_measurers = train_curr_measurers(conf, trainloader, curr_net, epoch=conf.epoch)

    curr_rec = curr_design(trainloader, net_cum_measurers)

    trainloader = dataSet(conf, curr_rec)
    adv_net = backboneSet(conf, type='DA')
    criterion, optimizer, scheduler = trainSet(conf, adv_net, l_lr=True)
    adv_net = train_da(conf, trainloader[0], adv_net, epoch=conf.epoch/2)
    criterion, optimizer, scheduler = trainSet(conf, adv_net)
    adv_net = train_da(conf, trainloader[1], adv_net, epoch=conf.epoch)

