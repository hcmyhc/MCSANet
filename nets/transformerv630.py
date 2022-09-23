# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor



#coding=utf-8

import math
import torch
import torch.nn.functional as F

##对FPN部分替换为Encoder，并使用卷积生成QKV

class DEPTHWISECONV(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DEPTHWISECONV, self).__init__()
        # 也相当于分组为1的分组卷积
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class DEPTHWISECONV7(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DEPTHWISECONV7, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=7,
                                    stride=1,
                                    padding=3,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class DEPTHWISECONV13(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DEPTHWISECONV13, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=13,
                                    stride=1,
                                    padding=6,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


# 构建SPP层(空间金字塔池化层)
class SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size() # num:样本数量 c:通道数 h:高 w:宽
        for i in range(self.num_levels):
            level = i+1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten


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




class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos1, memory, pos2,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                ):
        output = src

        # if len(output.shape) == 4:
        #     output = output.flatten(2).permute(2, 0, 1)
        if memory != None and len(memory.shape) == 4:
            memory = memory.flatten(2).permute(2, 0, 1)
        for layer in self.layers:
            output = layer(output, pos1, memory, pos2, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output




class QK2Attention(nn.Module):
    def __init__(self,d_model,nhead,height,activation = "relu"):
        super().__init__()
        self.head = nhead
        self.conv0 = nn.Linear((d_model // nhead) * 3 , height*height)
        self.conv1 = DEPTHWISECONV(d_model//nhead, d_model//nhead)
        self.batch1 = nn.InstanceNorm2d(d_model//nhead)

        self.conv2 = DEPTHWISECONV(d_model//nhead, (d_model//nhead) *2)

        self.batch2 = nn.InstanceNorm2d((d_model//nhead) *2)

        self.activation =  nn.LeakyReLU(0.1)
        self.norm = nn.LayerNorm(height * height)


        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
    def forward(self,out):

        hw,bs,ch= out.shape
        h= w = int(math.sqrt(hw))
        # temp = out.permute(1,2,0).reshape(bs, self.head, ch // self.head, hw)
        temp = out.permute(1,2,0).reshape(bs, ch, h, w)
        # out = out.permute(1,2,0).reshape(bs, ch, h, w)
        a,b = torch.split(temp,ch // self.head,dim=1)

        a = self.conv1(a)
        a = self.batch1(a)

        b = self.conv2(b)
        b = self.batch2(b)

        temp = torch.cat((a, b), dim=1)
        temp = self.activation(temp)

        out = temp.flatten(2).permute(0, 2, 1)
        out = self.norm(self.conv0(out))
        out = self.activation(out)

        out = F.softmax(out, dim=-1)

        attn = F.dropout(out, p=0.1)
        return attn
class QK4Attention(nn.Module):
    def __init__(self,d_model,nhead,height,activation = "relu"):
        super().__init__()
        self.head = nhead
        # self.conv = nn.Linear(d_model,nhead*d_model)
        self.conv0 = nn.Linear(d_model // nhead // 4 * 9, height*height)

        # self.conv1 = DEPTHWISECONV(d_model // nhead, (d_model // nhead) // 2)
        self.conv1 = DEPTHWISECONV(d_model// nhead , (d_model // nhead) )
        self.batch1 = nn.InstanceNorm2d((d_model // nhead) )

        self.conv2 = DEPTHWISECONV7(d_model // nhead, d_model // nhead // 2)
        self.batch2 = nn.InstanceNorm2d(d_model // nhead // 2)

        self.conv3 = DEPTHWISECONV7(d_model// nhead, (d_model//nhead) // 2)
        self.batch3 = nn.InstanceNorm2d((d_model//nhead) // 2)

        self.conv4 = DEPTHWISECONV13(d_model// nhead, (d_model//nhead) // 4)
        self.batch4 = nn.InstanceNorm2d((d_model//nhead) // 4)




        self.activation =  nn.LeakyReLU(0.1)
        self.norm = nn.LayerNorm(height * height)
        # self.se = SELayer(d_model)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
    def forward(self,out):

        hw,bs,ch= out.shape
        h= w = int(math.sqrt(hw))
        # new
        # temp = self.conv(out.permute(1, 0, 2)).permute(0, 2, 1).reshape(bs, ch * self.head, h, w)

        # temp = out.permute(1,2,0).reshape(bs, self.head, ch // self.head, hw)
        #old
        temp = out.permute(1,2,0).reshape(bs, ch, h, w)
        # out = out.permute(1,2,0).reshape(bs, ch, h, w)
        #old
        a,b,c,d = torch.split(temp,ch // self.head,dim=1)
        #new
        # a,b,c,d = torch.split(temp,ch,dim=1)
        # a = temp[:, 0, :, :].reshape(bs,ch//self.head,h,w)
        # b = temp[:, 1, :, :].reshape(bs,ch//self.head,h,w)
        # c = temp[:, 2, :, :].reshape(bs,ch//self.head,h,w)
        # d = temp[:, 3, :, :].reshape(bs,ch//self.head,h,w)
        a = self.conv1(a)
        a = self.batch1(a)
        # a = self.activation(a)
        b = self.conv2(b)
        b = self.batch2(b)
        # b = self.activation(b)
        c = self.conv3(c)
        c = self.batch3(c)
        # c = self.activation(c)
        d = self.conv4(d)
        d = self.batch4(d)
        # d = self.activation(d)
        temp = torch.cat((a, b, c, d), dim=1)
        # temp = self.batch1(temp)
        temp = self.activation(temp)
        # temp = self.batch1(temp)
        # temp = F.relu(temp)
        # temp = torch.cat((a, b), dim=1)
        # out = self.se(out)
        # 4月11晚上修改
        # out = out + temp
        # out = out * temp
        out = temp.flatten(2).permute(0, 2, 1)
        out = self.norm(self.conv0(out))
        out = self.activation(out)
        # mask = self.buildattnmask(bs, 4, 512, src_key_padding_mask)
        #4月11晚上修改
        # out = out.permute(0,2,1)  # +mask
        # out = out.flatten(2)  # +mask
        out = F.softmax(out, dim=-1)

        attn = F.dropout(out, p=0.1)
        return attn

class QK6Attention(nn.Module):
    def __init__(self,d_model,nhead,height,activation = "relu"):
        super().__init__()
        self.head = nhead
        # self.conv = nn.Linear(d_model,nhead*d_model)
        self.conv0 = nn.Linear(d_model // nhead // 2 * 9, height*height)



        # self.conv1 = DEPTHWISECONV(d_model//nhead, (d_model//nhead) *2)
        self.conv1 = DEPTHWISECONV(d_model// nhead, (d_model//nhead) )
        self.batch1 = nn.InstanceNorm2d((d_model//nhead) )

        # self.conv2 = DEPTHWISECONV(d_model//nhead, (d_model//nhead) *4)
        self.conv2 = DEPTHWISECONV7(d_model// nhead, (d_model//nhead) // 2)
        self.batch2 = nn.InstanceNorm2d((d_model//nhead) // 2)

        # self.conv3 = DEPTHWISECONV(d_model // nhead, (d_model // nhead) // 2)
        self.conv3 = DEPTHWISECONV7(d_model// nhead , (d_model // nhead) // 2)
        self.batch3 = nn.InstanceNorm2d((d_model // nhead) // 2)

        self.conv4 = DEPTHWISECONV13(d_model // nhead, (d_model // nhead) // 4)
        self.batch4 = nn.InstanceNorm2d((d_model // nhead) // 4)

        self.conv5 = DEPTHWISECONV(d_model// nhead , (d_model // nhead) )
        self.batch5 = nn.InstanceNorm2d((d_model // nhead) )

        self.conv6 = DEPTHWISECONV7(d_model// nhead , (d_model // nhead) // 2)
        self.batch6 = nn.InstanceNorm2d((d_model // nhead) // 2 )

        self.conv7 = DEPTHWISECONV7(d_model // nhead, (d_model // nhead) // 2)
        self.batch7 = nn.InstanceNorm2d((d_model // nhead) // 2)

        self.conv8 = DEPTHWISECONV13(d_model // nhead, (d_model // nhead) // 4)
        self.batch8 = nn.InstanceNorm2d((d_model // nhead) // 4)




        self.activation =  nn.LeakyReLU(0.1)
        self.norm = nn.LayerNorm(height * height)
        # self.se = SELayer(d_model)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
    def forward(self,out):

        hw,bs,ch= out.shape
        h= w = int(math.sqrt(hw))
        temp = out.permute(1, 2, 0).reshape(bs, ch, h, w)
        # temp = self.conv(out.permute(1,0,2))
        # temp = out.permute(1,2,0).reshape(bs, self.head, ch // self.head, hw)
        # temp = temp.permute(0,2,1).reshape(bs, ch*self.head, h, w)
        # out = out.permute(1,2,0).reshape(bs, ch, h, w)
        a,b,c,d,e,f,g,h = torch.split(temp,ch //self.head,dim=1)
        # a = temp[:, 0, :, :].reshape(bs,ch//self.head,h,w)
        # b = temp[:, 1, :, :].reshape(bs,ch//self.head,h,w)
        # c = temp[:, 2, :, :].reshape(bs,ch//self.head,h,w)
        # d = temp[:, 3, :, :].reshape(bs,ch//self.head,h,w)
        a = self.conv1(a)
        a = self.batch1(a)
        # a = self.activation(a)
        b = self.conv2(b)
        b = self.batch2(b)
        # b = self.activation(b)
        c = self.conv3(c)
        c = self.batch3(c)
        # c = self.activation(c)
        d = self.conv4(d)
        d = self.batch4(d)
        # d = self.activation(d)
        e = self.conv5(e)
        e = self.batch5(e)

        f = self.conv6(f)
        f = self.batch6(f)

        g = self.conv7(g)
        g = self.batch7(g)

        h = self.conv8(h)
        h = self.batch8(h)
        temp = torch.cat((a, b, c, d, e, f, g, h), dim=1)
        # temp = self.batch1(temp)
        temp = self.activation(temp)
        # temp = self.batch1(temp)
        # temp = F.relu(temp)
        # temp = torch.cat((a, b), dim=1)
        # out = self.se(out)
        # 4月11晚上修改
        # out = out + temp
        # out = out * temp
        out = temp.flatten(2).permute(0, 2, 1)
        out = self.norm(self.conv0(out))
        out = self.activation(out)
        # mask = self.buildattnmask(bs, 4, 512, src_key_padding_mask)
        #4月11晚上修改
        # out = out.permute(0,2,1)  # +mask
        # out = out.flatten(2)  # +mask
        out = F.softmax(out, dim=-1)

        attn = F.dropout(out, p=0.1)
        return attn

class TransformerEncoderBlock(nn.Module):
    def __init__(self,d_model, nhead,height,dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # self.encoderlayer0 = TransformerEncoderQKLayer(d_model,nhead,height)
        self.encoderlayer0 = TransformerEncoderQKLayer(d_model,nhead,height)
        # self.encoderlayer1 = TransformerEncoderLayer(d_model,nhead)
        # self.encoderlayerqk = TransformerEncoderQKLayer(d_model,nhead,height)
        self.encoderlayerqk = TransformerEncoderQKLayer(d_model,nhead,height)
    def forward(self,src, pos1, memory, pos2,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,):
        src = self.encoderlayer0(src,pos1, memory, pos2,src_mask,src_key_padding_mask)
        src = self.encoderlayerqk(src,pos1, memory, pos2,src_mask,src_key_padding_mask)
        return src

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, height,dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()


        # self.qk = QKAttention(d_model,nhead)
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.qk = QKAttention(d_model, nhead, height)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def buildattnmask(self,bsz,num_heads,src_len,key_padding_mask):

        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            key_padding_mask = key_padding_mask.to(torch.bool)
        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)

        attn_mask = key_padding_mask

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        return attn_mask

    def forward(self, src, pos1, memory, pos2,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,):
        #3月30日修改
        if memory != None and pos2 != None:
            if len(src.shape) == 4:
                src = src.flatten(2).permute(2, 0, 1)
            pos_embed1 = pos1.flatten(2).permute(2, 0, 1)
            pos_embed2 = pos2.flatten(2).permute(2, 0, 1)
            q = k = self.with_pos_embed(src, pos_embed1)
            src2 = self.self_attn(q, self.with_pos_embed(memory, pos_embed2) + k, value=memory + src, attn_mask=src_mask,
                           key_padding_mask=src_key_padding_mask)[0]
        else:
            if len(src.shape) == 4:
                src = src.flatten(2).permute(2,0,1)

            pos_embed1 = pos1.flatten(2).permute(2, 0, 1)
            q = k = self.with_pos_embed(src, pos_embed1)
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)[0]
        # hw bs c
        # src = src.flatten(2).permute(2,0,1)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


    # def forward(self, src,
    #             src_mask: Optional[Tensor] = None,
    #             src_key_padding_mask: Optional[Tensor] = None,
    #             pos: Optional[Tensor] = None):
    #     if self.normalize_before:
    #         return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
    #     return self.forward_post(src, src_mask, src_key_padding_mask, pos)
class TransformerEncoderQKLayer(nn.Module):

    def __init__(self, d_model,qka, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.qk = qka
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos



    def forward(self, src, pos1, memory, pos2,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,):
        #3月30日修改
        if memory != None :
            if len(src.shape) == 4:
                src = src.flatten(2).permute(2, 0, 1)
            if pos1 != None:
                pos_embed1 = pos1.flatten(2).permute(2, 0, 1)
            # pos_embed2 = pos2.flatten(2).permute(2, 0, 1)
                src = self.with_pos_embed(src, pos_embed1)
            attn = self.qk(src)
            src2 = torch.bmm(attn, memory.permute(1, 0, 2)).permute(1, 0, 2)
            # src2 = attn.permute(1, 0, 2)
            # src2 = self.self_attn(tgt, self.with_pos_embed(memory, pos_embed2), value=memory, attn_mask=src_mask,
            #                       key_padding_mask=src_key_padding_mask)[0]
        else:
            if len(src.shape) == 4:
                src = src.flatten(2).permute(2,0,1)
            v = src.permute(1,0,2)
            if pos1 != None:
                pos_embed1 = pos1.flatten(2).permute(2, 0, 1)
                src = self.with_pos_embed(src, pos_embed1)
            attn = self.qk(src)
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            src2 = torch.bmm(attn,v).permute(1,0,2)
            # src2 = attn.permute(1,0,2)
            # q = k = self.with_pos_embed(src, pos)
            # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
            #                       key_padding_mask=src_key_padding_mask)[0]
        # hw bs c
        # src = src.flatten(2).permute(2,0,1)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src




def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])




def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.leaky_relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
