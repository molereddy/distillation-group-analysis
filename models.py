import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.distributions import Categorical

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class NewResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, all_planes=[64, 128, 256, 512], adaptive_pool=False):
        super(NewResNet, self).__init__()
        self.in_planes = all_planes[0] #64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, all_planes[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, all_planes[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, all_planes[3], num_blocks[3], stride=2)
        self.linear = nn.Linear( all_planes[3] *block.expansion, num_classes)

        self.adaptive_pool = adaptive_pool
        if self.adaptive_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.conv_channels = [ all_planes[2]  ]
        self.xchannels   = [ all_planes[3] *block.expansion ] 

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_message(self):
        return 'EfficientNetV2_s (CIFAR)'#self.message

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        ft = [ out ]

        out = self.layer4(out)

        if self.adaptive_pool:
            out = self.avg_pool(out)
        else:
            out = F.avg_pool2d(out, 4)
            
        out = out.view(out.size(0), -1)
        features  = out 

        out = self.linear(out)
        logits = out
        #return out
        return features, logits, ft

def ResNet10_l(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, all_planes=[32, 64, 128, 256], adaptive_pool=adaptive_pool)


def ResNet10_m(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, all_planes=[16, 32, 64, 128], adaptive_pool=adaptive_pool)

def ResNet10_s2(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, all_planes=[16, 16, 32, 64], adaptive_pool=adaptive_pool)

def ResNet10_s(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, all_planes=[8, 16, 32, 64], adaptive_pool=adaptive_pool)

def ResNet10_xs(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, all_planes=[8, 16, 16, 32], adaptive_pool=adaptive_pool)

def ResNet10_xxs(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, all_planes=[8, 8, 16, 16], adaptive_pool=adaptive_pool)

def ResNet10_xxxs(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, all_planes=[4, 8, 8, 16], adaptive_pool=adaptive_pool)

def ResNet10(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, adaptive_pool=adaptive_pool)

def ResNet18(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, adaptive_pool=adaptive_pool)

def ResNet34(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, adaptive_pool=adaptive_pool)

def ResNet50(num_classes=100, adaptive_pool=False):
    return NewResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, adaptive_pool=adaptive_pool)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class SqueezeExcite(nn.Module):
    """ Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family
    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
            self, in_chs, rd_ratio=0.25, rd_channels=None, act_layer=nn.SiLU,
            gate_layer=nn.Sigmoid, force_act_layer=None, rd_round_fn=None):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = nn.SiLU(inplace=True)   #create_act_layer(act_layer, inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = nn.Sigmoid() #create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class IR_Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(IR_Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.se = SqueezeExcite( planes )

        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )


    def forward(self, x):
        out = F.silu(self.bn1(self.conv1(x)))
        out = F.silu(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class EfficientNet(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 240, 1, 1)]

    def __init__(self, num_classes=10):
        super(EfficientNet, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(240, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

        self.conv_channels = [240]
        self.xchannels   = [1280] 
        for m in [self.linear]:
            m.apply(init_weights)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(IR_Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def get_message(self):
        return 'Custom EfficientNet (CIFAR)'#self.message

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        ft =  [ out ]

        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        features  = out 

        out = self.linear(out)

        logits = out 
        return features, logits, ft
        #return out

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNetV2, self).__init__()

        self.conv_channels = [256]
        self.img_size = 224 #384 
        self.model = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=num_classes )
        #self.img_size = 224 
        #self.model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=num_classes )
        #for param in self.model.parameters():
        #    param.requires_grad = False
        #for param in self.model.classifier.parameters():
        #    param.requires_grad = True    

    def get_message(self):
        return 'EfficientNetV2_s (CIFAR)'#self.message

    def forward(self, x):
        x = F.interpolate( x, self.img_size )
        x = self.model.forward_features(x)
        ft = [ x ]

        x = self.model.global_pool(x)
        features = x
        x = self.model.classifier(x)
        logits = x
        return features, logits, ft 

class L_EfficientNet(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 3, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 4, 2),
           (6,  64, 5, 2),
           (6, 96, 4, 1),
           (6, 160, 4, 2),
           (6, 320, 2, 1)]

    def __init__(self, num_classes=10):
        super(L_EfficientNet, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

        self.conv_channels = [320]
        self.xchannels   = [1280] 
        for m in [self.linear]:
            m.apply(init_weights)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(IR_Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def get_message(self):
        return 'Custom EfficientNet (CIFAR)'#self.message

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        ft = [ out ]
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        features  = out 

        out = self.linear(out)

        logits = out 
        return features, logits, ft
        #return out

class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

        for m in [self.linear]:
            m.apply(init_weights)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def get_cifar_models(num_classes, model_name="", extra_path=None):
  #print(model_name)
  if model_name == 'efficientnet':
      return EfficientNet(num_classes)  

  if model_name == 'l_efficientnet':
      return L_EfficientNet(num_classes)  

  if model_name == 'efficientnetv2_s':
      return EfficientNetV2(num_classes)  

  adaptive_pool = config.dataset == 'tiny-imagenet-200'

  if model_name == 'ResNet50':
      return ResNet50(num_classes, adaptive_pool=adaptive_pool )  

  if model_name == 'ResNet34':
      return ResNet34(num_classes, adaptive_pool=adaptive_pool )  

  if model_name == 'ResNet18':
      return ResNet18(num_classes, adaptive_pool=adaptive_pool )  

  if model_name == 'ResNet10':
      return ResNet10(num_classes, adaptive_pool=adaptive_pool )  

  if model_name == 'ResNet10_l':
      return ResNet10_l(num_classes, adaptive_pool=adaptive_pool )  

  if model_name == 'ResNet10_m':
      return ResNet10_m(num_classes, adaptive_pool=adaptive_pool )  

  if model_name == 'ResNet10_s':
      return ResNet10_s(num_classes, adaptive_pool=adaptive_pool )  

  if model_name == 'ResNet10_xs':
      return ResNet10_xs(num_classes, adaptive_pool=adaptive_pool )  

  if model_name == 'ResNet10_xxs':
      return ResNet10_xxs(num_classes, adaptive_pool=adaptive_pool )  

  if model_name == 'ResNet10_xxxs':
      return ResNet10_xxxs(num_classes, adaptive_pool=adaptive_pool )  

  raise ValueError('invalid model-name : {:}'.format(model_name))


    
