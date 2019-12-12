from .modelzoo import *
from .utils.include import *


######################################################################################################333

# http://wuhuikai.me/FastFCNProject/fast_fcn.pdf
# FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation
# https://github.com/wuhuikai/FastFCN

class ASPPConv(nn.Module):
    def __init__(self, in_channel, out_channel, dilation):
        super(ASPPConv, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.module(x)
        return x

class ASPPPool(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ASPPPool, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size,C,H,W = x.shape
        x = self.module(x)
        x = F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel=256,rate=[6,12,18], dropout_rate=0):
        super(ASPP, self).__init__()

        self.atrous0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.atrous1 = ASPPConv(in_channel, out_channel, rate[0])
        self.atrous2 = ASPPConv(in_channel, out_channel, rate[1])
        self.atrous3 = ASPPConv(in_channel, out_channel, rate[2])
        self.atrous4 = ASPPPool(in_channel, out_channel)

        self.combine = nn.Sequential(
            nn.Conv2d(5 * out_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):

        x = torch.cat([
            self.atrous0(x),
            self.atrous1(x),
            self.atrous2(x),
            self.atrous3(x),
            self.atrous4(x),
        ],1)
        x = self.combine(x)
        return x


#------
def resize_like(x, reference, mode='bilinear'):
    if x.shape[2:] !=  reference.shape[2:]:
        if mode=='bilinear':
            x = F.interpolate(x, size=reference.shape[2:],mode='bilinear',align_corners=False)
        if mode=='nearest':
            x = F.interpolate(x, size=reference.shape[2:],mode='nearest')
    return x

def fuse(x, mode='cat'):
    batch_size,C0,H0,W0 = x[0].shape

    for i in range(1,len(x)):
        _,_,H,W = x[i].shape
        if (H,W)!=(H0,W0):
            x[i] = F.interpolate(x[i], size=(H0,W0), mode='bilinear', align_corners=False)

    if mode=='cat':
        return torch.cat(x,1)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, dilation, groups=in_channel, bias=bias)
        self.bn   = nn.BatchNorm2d(in_channel)
        self.pointwise = nn.Conv2d(in_channel, out_channel, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


#JPU
class JointPyramidUpsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(JointPyramidUpsample, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel[0], out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel[1], out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel[2], out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        #-------------------------------

        self.dilation0 = nn.Sequential(
            SeparableConv2d(3*out_channel, out_channel, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dilation1 = nn.Sequential(
            SeparableConv2d(3*out_channel, out_channel, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dilation2 = nn.Sequential(
            SeparableConv2d(3*out_channel, out_channel, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dilation3 = nn.Sequential(
            SeparableConv2d(3*out_channel, out_channel, kernel_size=3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x0 = self.conv0(x[0])
        x1 = self.conv1(x[1])
        x2 = self.conv2(x[2])

        x0 = resize_like(x0, x2, mode='nearest')
        x1 = resize_like(x1, x2, mode='nearest')
        x = torch.cat([x0,x1,x2], dim=1)

        d0 = self.dilation0(x)
        d1 = self.dilation1(x)
        d2 = self.dilation2(x)
        d3 = self.dilation3(x)
        x = torch.cat([d0,d1,d2,d3], dim=1)
        return x


class ConvGnUp2d(nn.Module):
    def __init__(self, in_channel, out_channel, num_group=32, kernel_size=3, padding=1, stride=1):
        super(ConvGnUp2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.gn   = nn.GroupNorm(num_group, out_channel)

    def forward(self,x):
        x = self.conv(x)
        x = self.gn(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x

def upsize_add(x, lateral):
    return F.interpolate(x, size=lateral.shape[2:], mode='nearest') + lateral

def upsize(x, scale_factor=2):
    x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x


class Net(nn.Module):

    def __init__(self, name="ResNet34", num_class=4):
        super(Net, self).__init__()

        if name == "ResNet34":
            self.basemodel = resnet34(True)
            self.planes = [256 // 4, 512 // 4, 1024 // 4, 2048 // 4]
            
        self.down = True
        
        if name == 'seresnext50':
            self.basemodel = se_resnext50_32x4d(pretrained='imagenet')
            self.planes = [256 // 4, 512 // 4, 1024 // 4, 2048 // 4]
            inplanes = 64
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
            layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                        ceil_mode=True)))
            self.basemodel.layer0 = nn.Sequential(OrderedDict(layer0_modules))

            self.down1 = nn.Conv2d(256, self.planes[0], kernel_size=1)
            self.down2 = nn.Conv2d(512, self.planes[1], kernel_size=1)
            self.down3 = nn.Conv2d(1024, self.planes[2], kernel_size=1)
            self.down4 = nn.Conv2d(2048, self.planes[3], kernel_size=1)
            
        if name == 'seresnext26':
            self.basemodel = seresnext26_32x4d(pretrained=True)
            self.planes = [256 // 4, 512 // 4, 1024 // 4, 2048 // 4]
            self.startconv = nn.Conv2d(3, 3, kernel_size=1)

            self.down1 = nn.Conv2d(256, self.planes[0], kernel_size=1)
            self.down2 = nn.Conv2d(512, self.planes[1], kernel_size=1)
            self.down3 = nn.Conv2d(1024, self.planes[2], kernel_size=1)
            self.down4 = nn.Conv2d(2048, self.planes[3], kernel_size=1)
        if name == 'seresnext101':
            self.basemodel = se_resnext101_32x4d(pretrained='imagenet')
            self.planes = [256 // 4, 512 // 4, 1024 // 4, 2048 // 4]
            inplanes = 64
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
            layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                        ceil_mode=True)))
            self.basemodel.layer0 = nn.Sequential(OrderedDict(layer0_modules))

            self.down1 = nn.Conv2d(256, self.planes[0], kernel_size=1)
            self.down2 = nn.Conv2d(512, self.planes[1], kernel_size=1)
            self.down3 = nn.Conv2d(1024, self.planes[2], kernel_size=1)
            self.down4 = nn.Conv2d(2048, self.planes[3], kernel_size=1)
        
        if name == 'dpn68':
            self.startconv = nn.Conv2d(3, 3, kernel_size=1)
            self.basemodel = dpn68(pretrained=True)

            self.planes = [256 // 4, 512 // 4, 1024 // 4, 2048 // 4]
            self.down1 = nn.Conv2d(144, self.planes[0], kernel_size=1)
            self.down2 = nn.Conv2d(320, self.planes[1], kernel_size=1)
            self.down3 = nn.Conv2d(704, self.planes[2], kernel_size=1)
            self.down4 = nn.Conv2d(832, self.planes[3], kernel_size=1)
            
        if name == 'efficientnet-b7':
            self.startconv = nn.Conv2d(3, 3, kernel_size=1)
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b7')
            self.planes = [48, 48, 80, 160]
            self.down = False
            
        if name == 'efficientnet-b5':
            self.startconv = nn.Conv2d(3, 3, kernel_size=1)
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b5')
            self.planes = [40, 40, 128, 176]
            self.down = False
            
        if name == 'efficientnet-b4':
            self.startconv = nn.Conv2d(3, 3, kernel_size=1)
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b4')
            self.planes = [32, 56, 112, 272]
            self.down = False

        if name == 'efficientnet-b3':
            self.startconv = nn.Conv2d(3, 3, kernel_size=1)
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b3')

            self.planes = [32, 48, 136, 232]
            self.down = False

        if name == 'efficientnet-b2':
            self.startconv = nn.Conv2d(3, 3, kernel_size=1)
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b2')

            self.planes = [24, 48, 120, 352]
            self.down = False

        if name == 'efficientnet-b1':
            self.startconv = nn.Conv2d(3, 3, kernel_size=1)
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b1')

            self.planes = [24, 40, 112, 320]
            self.down = False
            
        
        self.Upsample = nn.Sequential(
            ConvGnUp2d(128, 128),
            ConvGnUp2d(128, 64),
        )

        self.jpu = JointPyramidUpsample([self.planes[3], self.planes[2], self.planes[1], self.planes[0]], self.planes[0])
        self.aspp = ASPP(self.planes[0] * 4, 128, rate=[4, 8, 12], dropout_rate=0.1)
        self.logit = nn.Conv2d(64, num_class, kernel_size=1)


    def forward(self, x):
        batch_size, C, H, W = x.shape
        x = F.pad(x,[18,18,2,2],mode='constant', value=0) #pad = (left, right, top, down)
        x1, x2, x3, x4 = self.basemodel(x)
        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        
        if self.down:
            x1 = self.down1(x1)
            x2 = self.down2(x2)
            x3 = self.down3(x3)
            x4 = self.down4(x4)
        
        x = self.jpu([x4, x3, x2, x1])
        x = self.aspp(x)
        x = self.Upsample(x)
        logit = self.logit(x)
        logit = self.logit(x)[:,:,1:350+1,10:525+10]

        #---
        # probability_mask  = torch.sigmoid(logit)
        # probability_label = F.adaptive_max_pool2d(probability_mask,1).view(batch_size,-1)
        
        return logit


