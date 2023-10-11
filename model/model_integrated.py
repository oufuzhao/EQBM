import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function
import torch.utils.data
from torch import nn
from torch.autograd import Function
from torch.nn import Sequential


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Adv_Model(nn.Module):
    def __init__(self, base_backbone, out_size=512, pretrained=False, type='DA'):
        super(Adv_Model, self).__init__()
        self.out_size = out_size
        self.pretrained = pretrained
        self.base_backbone = base_backbone
        self.type = type

        self.main_classifier = Sequential(                                                   
                                nn.Linear(self.out_size, 256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(True),
                                nn.Dropout(0.5),
                                nn.Linear(256, 256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(True),
                                nn.Linear(256, 5),
                                nn.Softmax(dim=1)
                                )

        if self.type == 'Curr_Measurers':                                                      
            self.uncertainty_domain_classifier_1 = Sequential(
                                nn.Linear(self.out_size, 256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(True),
                                )
            self.uncertainty_domain_classifier_2 = Sequential(
                                nn.Linear(256, 2, bias=False),
                                )
            self.uncertainty_domain_classifier_3 = Sequential(
                                nn.LogSoftmax(dim=1)
                                )

        if self.type == 'DA' or self.type == 'Test':                                 
            self.rank_classifier = Sequential(
                                    nn.Linear(self.out_size, 256),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(True),
                                    nn.Dropout(0.5),
                                    nn.Linear(256, 256),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(True),
                                    nn.Linear(256, 2),
                                    nn.LogSoftmax(dim=1)
                                    )
    
            self.adv_domain_classifier = Sequential(
                                    nn.Linear(self.out_size, 256),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(True),
                                    nn.Linear(256, 2),
                                    # nn.Softmax(dim=1)
                                    nn.LogSoftmax(dim=1)
                                    )

    def forward(self, input_data1, input_data2=None,  alpha=1.0, train=True):

        if self.type == 'Curr_Measurers':
            if train:
                feature1 = self.base_backbone(input_data1)
                feature2 = self.base_backbone(input_data2)
                main_classifier_output = self.main_classifier(feature1)
                uncertainty_domain_classifier_output1_1 = self.uncertainty_domain_classifier_1(feature1)
                uncertainty_domain_classifier_output2_1 = self.uncertainty_domain_classifier_2(uncertainty_domain_classifier_output1_1)
                uncertainty_domain_classifier_output3_1 = self.uncertainty_domain_classifier_3(uncertainty_domain_classifier_output2_1)
                uncertainty_domain_classifier_output1_2 = self.uncertainty_domain_classifier_1(feature2)
                uncertainty_domain_classifier_output2_2 = self.uncertainty_domain_classifier_2(uncertainty_domain_classifier_output1_2)
                uncertainty_domain_classifier_output3_2 = self.uncertainty_domain_classifier_3(uncertainty_domain_classifier_output2_2)
                return main_classifier_output, uncertainty_domain_classifier_output3_1, uncertainty_domain_classifier_output3_2
            else:
                feature1 = self.base_backbone(input_data1)
                main_classifier_output1 = self.main_classifier(feature1)
                uncertainty_domain_classifier_output1_1 = self.uncertainty_domain_classifier_1(feature1)
                return main_classifier_output1, uncertainty_domain_classifier_output1_1

        elif self.type == 'DA' or self.type == 'Test':
            if train:
                feature1 = self.base_backbone(input_data1)
                feature2 = self.base_backbone(input_data2)
                main_classifier_output1 = self.main_classifier(feature1)
                main_classifier_output2 = self.main_classifier(feature2)
                rank_classifier_output = self.rank_classifier(feature1 - feature2)
                reverse_feature1 = ReverseLayerF.apply(feature1, alpha)
                adv_domain_classifier_output1 = self.adv_domain_classifier(reverse_feature1)
                reverse_feature2 = ReverseLayerF.apply(feature2, alpha)
                adv_domain_classifier_output2 = self.adv_domain_classifier(reverse_feature2)
                return main_classifier_output1, main_classifier_output2, rank_classifier_output, adv_domain_classifier_output1, adv_domain_classifier_output2
            else:
                feature1 = self.base_backbone(input_data1)
                main_classifier_output1 = self.main_classifier(feature1)
                return main_classifier_output1
