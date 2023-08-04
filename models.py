from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

model_attributes = {
    "bert": {
        "feature_type": "text"
    },
    'inception_v3': {
        'feature_type': 'image',
        'target_resolution': (299, 299),
        'flatten': False
    },
    'wideresnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet18': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet34': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': False
    },
    'raw_logistic_regression': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': True,
    },
    "bert-base-uncased": {
        'feature_type': 'text'
    }
}

class SimKD(nn.Module):
    """CVPR-2022: Knowledge Distillation with the Reused Teacher Classifier"""
    def __init__(self, *, s_n, t_n, factor=2): 
        super(SimKD, self).__init__()
       
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))       

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1, groups=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups)
        
        # A bottleneck design to reduce extra parameters
        setattr(self, 'transfer', nn.Sequential(
            conv1x1(s_n, t_n//factor),
            nn.BatchNorm2d(t_n//factor),
            nn.ReLU(inplace=True),
            conv3x3(t_n//factor, t_n//factor),
            # depthwise convolution
            #conv3x3(t_n//factor, t_n//factor, groups=t_n//factor),
            nn.BatchNorm2d(t_n//factor),
            nn.ReLU(inplace=True),
            conv1x1(t_n//factor, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
            ))
        
    def forward(self, feat_s, feat_t, cls_t):
        
        # Spatial Dimension Alignment
        s_H, t_H = feat_s.shape[2], feat_t.shape[2]
        if s_H > t_H:
            source = F.adaptive_avg_pool2d(feat_s, (t_H, t_H))
            target = feat_t
        else:
            source = feat_s
            target = F.adaptive_avg_pool2d(feat_t, (s_H, s_H))
        
        trans_feat_t=target
        
        # Channel Alignment
        trans_feat_s = getattr(self, 'transfer')(source)

        # Prediction via Teacher Classifier
        temp_feat = self.avg_pool(trans_feat_s)
        temp_feat = temp_feat.view(temp_feat.size(0), -1)
        pred_feat_s = cls_t(temp_feat)
        
        return trans_feat_s, trans_feat_t, pred_feat_s


class FeatResNet(nn.Module):
    
    def __init__(self, core_resnet):
        super(FeatResNet, self).__init__()
        self.internal = core_resnet
        self.fc = self.internal.fc
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.internal.conv1(x)
        x = self.internal.bn1(x)
        x = self.internal.relu(x)
        x = self.internal.maxpool(x)

        x = self.internal.layer1(x)
        x = self.internal.layer2(x)
        x = self.internal.layer3(x)
        x = self.internal.layer4(x)
        ft_full = x

        x = self.internal.avgpool(x)
        x = torch.flatten(x, 1)
        ft = x
        x = self.internal.fc(x)

        return [ft_full, ft], x