import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from .transformerv630 import TransformerEncoderQKLayer,QK2Attention,QK4Attention

#-------------------------------------------------#
#   卷积块
#   Conv2d + BatchNorm2d + LeakyReLU
#-------------------------------------------------#
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // rotio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_channels // rotio, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5,9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

         
'''
                    input
                      |
                  BasicConv
                      -----------------------
                      |                     |
                 route_group              route
                      |                     |
                  BasicConv                 |
                      |                     |
    -------------------                     |
    |                 |                     |
 route_1          BasicConv                 |
    |                 |                     |
    -----------------cat                    |
                      |                     |
        ----      BasicConv                 |
        |             |                     |
      feat           cat---------------------
                      |
                 MaxPooling2D
'''

class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()
        self.out_channels = out_channels

        self.conv1 = BasicConv(in_channels, out_channels, 3)

        self.conv2 = BasicConv(out_channels//2, out_channels//2, 3)
        self.conv3 = BasicConv(out_channels//2, out_channels//2, 3)

        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2,2],[2,2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1(x)
        # 引出一个大的残差边route
        route = x
        
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c//2, dim = 1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x,route1], dim = 1) 

        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim = 1)
        
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x,feat

class CSPDarkNet(nn.Module):
    def __init__(self):
        super(CSPDarkNet, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)

        self.encoder1 = TransformerEncoderQKLayer(d_model=64,qka=QK2Attention(64,2,52))
        self.encoder2 = TransformerEncoderQKLayer(d_model=128,qka=QK4Attention(128,4,26))
        self.encoder3 = TransformerEncoderQKLayer(d_model=256,qka=QK4Attention(256,4,13))
        self.downsample1 = PatchMerging((52, 52), dim=128)
        self.downsample2 = PatchMerging((26, 26), dim=256)
        # 104,104,64 -> 52,52,128
        self.resblock_body1 =  Resblock_body(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 =  Resblock_body(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 =  Resblock_body(256, 256)

        self.conv3_chan = ChannelAttention(64)
        self.conv3_spit = SpatialAttention()

        self.conv4 = BasicConv(512, 256, kernel_size=1)

        self.conv4_chan = ChannelAttention(256)
        self.conv4_spit = SpatialAttention()

        self.conv5 = BasicConv(1024, 512, kernel_size=1)
        self.conv5_chan = ChannelAttention(512)
        self.conv5_spit = SpatialAttention()

        self.num_features = 1



    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _    = self.resblock_body1(x)

        l1, x = torch.split(x, 64, dim=1)
        l1 = self.conv3_chan(l1)*l1
        l1 = self.conv3_spit(l1)*l1

        x = self.encoder1(x, None, None, None)
        x = torch.cat((l1.flatten(2).permute(2,0,1),x),dim=2)
        x = x.permute(1,2,0).reshape(-1,128,52,52)

        l2 = x

        l2,l2_t = self.resblock_body2(l2)
        l2 = self.conv4_chan(l2) * l2
        l2 = self.conv4_spit(l2) * l2


        x = self.downsample1(l2_t.flatten(2).permute(0, 2, 1)) #bs hw c
        t1,x = torch.split(x,128,2) # bs hw c//2
        x = self.encoder2(x.permute(1, 0, 2), None, None, None) # hw bs c//2
        x = torch.cat((t1,x.permute(1,0,2)),dim=2).permute(0,2,1).reshape(-1,256,26,26)

        x = torch.cat((l2,x),dim=1) # bs hw c
        x = self.conv4(x)
        l3 = x


        l3,feat1 = self.resblock_body3(l3)

        l3 = self.conv5_chan(l3) * l3
        l3 = self.conv5_spit(l3) * l3

        x = self.downsample2(feat1.flatten(2).permute(0, 2, 1))  # bs hw c
        t2,x  = torch.split(x, 256, 2)  # bs hw c//2
        x = self.encoder3(x.permute(1, 0, 2), None, None, None)  # hw bs c//2
        x = torch.cat((t2,x.permute(1, 0, 2)), dim=2).permute(0, 2, 1).reshape(-1, 512, 13, 13)
        x = torch.cat((l3, x), dim=1)
        feat2 = self.conv5(x)


        return feat1,feat2

def darknet53_tiny(pretrained, **kwargs):
    model = CSPDarkNet()
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model
