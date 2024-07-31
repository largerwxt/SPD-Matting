# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers

import ppmatting

__all__ = [
    "HRNet_W18_Small_V1", "HRNet_W18_Small_V2", "HRNet_W18", "HRNet_W30",
    "HRNet_W32", "HRNet_W40", "HRNet_W44", "HRNet_W48", "HRNet_W60", "HRNet_W64"
]


class HRNet(nn.Layer):
    """
    The HRNet implementation based on PaddlePaddle.

    The original article refers to
    Jingdong Wang, et, al. "HRNet：Deep High-Resolution Representation Learning for Visual Recognition"
    (https://arxiv.org/pdf/1908.07919.pdf).

    Args:
        pretrained (str, optional): The path of pretrained model.
        stage1_num_modules (int, optional): Number of modules for stage1. Default 1.
        stage1_num_blocks (list, optional): Number of blocks per module for stage1. Default (4).
        stage1_num_channels (list, optional): Number of channels per branch for stage1. Default (64).
        stage2_num_modules (int, optional): Number of modules for stage2. Default 1.
        stage2_num_blocks (list, optional): Number of blocks per module for stage2. Default (4, 4).
        stage2_num_channels (list, optional): Number of channels per branch for stage2. Default (18, 36).
        stage3_num_modules (int, optional): Number of modules for stage3. Default 4.
        stage3_num_blocks (list, optional): Number of blocks per module for stage3. Default (4, 4, 4).
        stage3_num_channels (list, optional): Number of channels per branch for stage3. Default [18, 36, 72).
        stage4_num_modules (int, optional): Number of modules for stage4. Default 3.
        stage4_num_blocks (list, optional): Number of blocks per module for stage4. Default (4, 4, 4, 4).
        stage4_num_channels (list, optional): Number of channels per branch for stage4. Default (18, 36, 72. 144).
        has_se (bool, optional): Whether to use Squeeze-and-Excitation module. Default False.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
    """

    def __init__(self,
                 input_channels=3,
                 pretrained=None,
                 stage1_num_modules=1,
                 stage1_num_blocks=(4, ),
                 stage1_num_channels=(64, ),
                 stage2_num_modules=1,
                 stage2_num_blocks=(4, 4),
                 stage2_num_channels=(18, 36),
                 stage3_num_modules=4,
                 stage3_num_blocks=(4, 4, 4),
                 stage3_num_channels=(18, 36, 72),
                 stage4_num_modules=3,
                 stage4_num_blocks=(4, 4, 4, 4),
                 stage4_num_channels=(18, 36, 72, 144),
                 has_se=False,   # SE模块
                 align_corners=False,
                 padding_same=True):
        super(HRNet, self).__init__()
        self.pretrained = pretrained
        self.stage1_num_modules = stage1_num_modules
        self.stage1_num_blocks = stage1_num_blocks
        self.stage1_num_channels = stage1_num_channels
        self.stage2_num_modules = stage2_num_modules
        self.stage2_num_blocks = stage2_num_blocks
        self.stage2_num_channels = stage2_num_channels
        self.stage3_num_modules = stage3_num_modules
        self.stage3_num_blocks = stage3_num_blocks
        self.stage3_num_channels = stage3_num_channels
        self.stage4_num_modules = stage4_num_modules
        self.stage4_num_blocks = stage4_num_blocks
        self.stage4_num_channels = stage4_num_channels
        self.has_se = has_se
        self.align_corners = align_corners

        self.feat_channels = [i for i in stage4_num_channels]  # [48,96,192,384]
        self.feat_channels = [64] + self.feat_channels  # [112,160,256,448]

        self.conv_layer1_1 = layers.ConvBNReLU(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1 if not padding_same else 'same',
            bias_attr=False)

        self.conv_layer1_2 = layers.ConvBNReLU(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1 if not padding_same else 'same',
            bias_attr=False)

        self.la1 = Layer1(
            num_channels=64,
            num_blocks=self.stage1_num_blocks[0],  # 4
            num_filters=self.stage1_num_channels[0],  # 64
            has_se=has_se,
            name="layer2",
            padding_same=padding_same)

        self.tr1 = TransitionLayer(
            in_channels=[self.stage1_num_channels[0] * 4],  # [64*4]  在之前每个分支的通道数
            out_channels=self.stage2_num_channels,  # [48,96]   完成transition之后每个分支的通道数
            name="tr1",
            padding_same=padding_same)

        self.st2 = Stage(
            num_channels=self.stage2_num_channels,  # [48,96]
            num_modules=self.stage2_num_modules,   # 1
            num_blocks=self.stage2_num_blocks,    # [4,4]
            num_filters=self.stage2_num_channels,  # [48,96]
            has_se=self.has_se,
            name="st2",
            align_corners=align_corners,
            padding_same=padding_same)

        self.tr2 = TransitionLayer(
            in_channels=self.stage2_num_channels,  # [48,96]
            out_channels=self.stage3_num_channels,  # [48,96,192]
            name="tr2",
            padding_same=padding_same)
        self.st3 = Stage(
            num_channels=self.stage3_num_channels,  # [48,96,192]
            num_modules=self.stage3_num_modules,   # 4
            num_blocks=self.stage3_num_blocks,    # [4,4,4]
            num_filters=self.stage3_num_channels,   # [48,96,192]
            has_se=self.has_se,
            name="st3",
            align_corners=align_corners,
            padding_same=padding_same)

        self.tr3 = TransitionLayer(
            in_channels=self.stage3_num_channels,  # [48,96,192]
            out_channels=self.stage4_num_channels,   # [48,96,192,384]
            name="tr3",
            padding_same=padding_same)
        self.st4 = Stage(
            num_channels=self.stage4_num_channels,   # [48,96,192,384]
            num_modules=self.stage4_num_modules,     # 3
            num_blocks=self.stage4_num_blocks,       # [4,4,4,4]
            num_filters=self.stage4_num_channels,    # [48,96,192,384]
            has_se=self.has_se,
            name="st4",
            align_corners=align_corners,
            padding_same=padding_same)

        self.init_weight()

    def forward(self, x):
        feat_list = []
        conv1 = self.conv_layer1_1(x)
        feat_list.append(conv1)
        conv2 = self.conv_layer1_2(conv1)

        la1 = self.la1(conv2)    # (h/4,w/4,64*4)

        tr1 = self.tr1([la1])    # [(h/4,w/4,48),(h/8,w/8,96)]
        st2 = self.st2(tr1)      # [(h/4,w/4,48),(h/8,w/8,96)]

        tr2 = self.tr2(st2)      # [(h/4,w/4,48),(h/8,w/8,96),(h/16,w/16,192)]
        st3 = self.st3(tr2)      # [(h/4,w/4,48),(h/8,w/8,96),(h/16,w/16,192)]

        tr3 = self.tr3(st3)      # [(h/4,w/4,48),(h/8,w/8,96),(h/16,w/16,192),(h/32,w/32,384)]
        st4 = self.st4(tr3)      # [(h/4,w/4,48),(h/8,w/8,96),(h/16,w/16,192),(h/32,w/32,384)]

        feat_list = feat_list + st4  # [(h/2,w/2,64),(h/4,w/4,48),(h/8,w/8,96),(h/16,w/16,192),(h/32,w/32,384)]

        return feat_list

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
        if self.pretrained is not None:
            ppmatting.utils.load_pretrained_model(self, self.pretrained)


class Layer1(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 num_blocks,
                 has_se=False,
                 name=None,
                 padding_same=True):
        super(Layer1, self).__init__()

        self.bottleneck_block_list = []

        for i in range(num_blocks):
            bottleneck_block = self.add_sublayer(
                "bb_{}_{}".format(name, i + 1),
                BottleneckBlock(
                    num_channels=num_channels if i == 0 else num_filters * 4,
                    num_filters=num_filters,
                    has_se=has_se,
                    stride=1,
                    downsample=True if i == 0 else False,
                    name=name + '_' + str(i + 1),
                    padding_same=padding_same))
            self.bottleneck_block_list.append(bottleneck_block)

    def forward(self, x):
        conv = x
        for block_func in self.bottleneck_block_list:
            conv = block_func(conv)
        return conv  # （h/4,w/4,64*4)


class TransitionLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, name=None, padding_same=True):
        super(TransitionLayer, self).__init__()

        num_in = len(in_channels)  # 1 计算之前有多少分支
        num_out = len(out_channels)  # 2 计算之后有多少分支
        self.conv_bn_func_list = []
        for i in range(num_out):  # [48,96]  i取值为 0，1
            residual = None
            # 我们可以直接利用branches_pre已有分支作为branches_cur的其中一个分支
            # 这个操作是hrnet的一个创新操作：在缩减特征图shape提取特征的同时，始终保留高分辨率特征图
            if i < num_in:  # i=0
                if in_channels[i] != out_channels[i]:  # 64*4!=48
                    # 如果branches_cur通道数=branches_pre通道数，那么这个分支直接就可以用，不用做任何变化
                    # 如果branches_cur通道数！=branches_pre通道数，那么就要用一个cnn网络改变通道数
                    # 注意这个cnn是不会改变特征图的shape
                    # 在stage1中，pre通道数是256，cur通道数为48，所以要添加这一层cnn改变通道数
                    residual = self.add_sublayer(  # add_sublayer方法：返回一个由所有子层组成的列表
                        "transition_{}_layer_{}".format(name, i + 1),   # transition_str1_layer_1
                        layers.ConvBNReLU(
                            in_channels=in_channels[i],  # 64*4
                            out_channels=out_channels[i],   # 48
                            kernel_size=3,
                            padding=1 if not padding_same else 'same',
                            bias_attr=False))
            else:
                # 我们必须要利用branches_pre里的分支无中生有一个新分支
                # 这就是常见的缩减图片shape，增加通道数提特征的操作
                residual = self.add_sublayer(
                    "transition_{}_layer_{}".format(name, i + 1),   # transition_str1_layer_2
                    layers.ConvBNReLU(
                        in_channels=in_channels[-1],  # 64*4
                        out_channels=out_channels[i],  # 96
                        kernel_size=3,
                        stride=2,
                        padding=1 if not padding_same else 'same',
                        bias_attr=False))
            self.conv_bn_func_list.append(residual)

    def forward(self, x):
        outs = []
        for idx, conv_bn_func in enumerate(self.conv_bn_func_list):
            if conv_bn_func is None:
                outs.append(x[idx])
            else:
                if idx < len(x):
                    outs.append(conv_bn_func(x[idx]))
                else:
                    outs.append(conv_bn_func(x[-1]))
        return outs


class Branches(nn.Layer):
    def __init__(self,
                 num_blocks,  # [4,4]
                 in_channels,  # [48,96]
                 out_channels,  # [48,96]
                 has_se=False,
                 name=None,
                 padding_same=True):
        super(Branches, self).__init__()

        self.basic_block_list = []  # [branches1[4],branches2[4]]

        for i in range(len(out_channels)):   # 2[0,1] i能取到0，1
            self.basic_block_list.append([])
            for j in range(num_blocks[i]):  # j能取到0，1，2，3
                in_ch = in_channels[i] if j == 0 else out_channels[i]  # 48/96
                basic_block_func = self.add_sublayer(
                    "bb_{}_branch_layer_{}_{}".format(name, i + 1, j + 1),
                    BasicBlock(
                        num_channels=in_ch,  # j=0，48
                        num_filters=out_channels[i],  # 48
                        has_se=has_se,
                        name=name + '_branch_layer_' + str(i + 1) + '_' +
                        str(j + 1),
                        padding_same=padding_same))
                self.basic_block_list[i].append(basic_block_func)

    def forward(self, x):
        outs = []
        for idx, input in enumerate(x):  # x=[(h/4,w/4,48),(h/8,w/8,96)]
            conv = input
            for basic_block_func in self.basic_block_list[idx]:
                conv = basic_block_func(conv)
            outs.append(conv)
        return outs  # [(h/4,w/4,48),(h/8,w/8,96)]


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 has_se,
                 stride=1,
                 downsample=False,
                 name=None,
                 padding_same=True):
        super(BottleneckBlock, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = layers.ConvBNReLU(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=1,
            bias_attr=False)

        self.conv2 = layers.ConvBNReLU(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=stride,
            padding=1 if not padding_same else 'same',
            bias_attr=False)

        self.conv3 = layers.ConvBN(
            in_channels=num_filters,
            out_channels=num_filters * 4,
            kernel_size=1,
            bias_attr=False)

        if self.downsample:               # 如果x与conv3输出图维度本来就相同，就意味着可以直接相加，那么downsample会为空，自然就不会进行下面操作
            self.conv_down = layers.ConvBN(
                in_channels=num_channels,
                out_channels=num_filters * 4,
                kernel_size=1,
                bias_attr=False)

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters * 4,
                num_filters=num_filters * 4,
                reduction_ratio=16,
                name=name + '_fc')

        self.add = layers.Add()
        self.relu = layers.Activation("relu")

    def forward(self, x):
        residual = x
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        if self.downsample:
            residual = self.conv_down(x)

        if self.has_se:
            conv3 = self.se(conv3)

        y = self.add(conv3, residual)
        y = self.relu(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride=1,
                 has_se=False,
                 downsample=False,
                 name=None,
                 padding_same=True):
        super(BasicBlock, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = layers.ConvBNReLU(
            in_channels=num_channels,  # 48
            out_channels=num_filters,  # 48
            kernel_size=3,
            stride=stride,
            padding=1 if not padding_same else 'same',
            bias_attr=False)
        self.conv2 = layers.ConvBN(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            padding=1 if not padding_same else 'same',
            bias_attr=False)

        if self.downsample:
            self.conv_down = layers.ConvBNReLU(
                in_channels=num_channels,
                out_channels=num_filters,
                kernel_size=1,
                bias_attr=False)

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters,
                num_filters=num_filters,
                reduction_ratio=16,
                name=name + '_fc')

        self.add = layers.Add()
        self.relu = layers.Activation("relu")

    def forward(self, x):
        residual = x
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        if self.downsample:
            residual = self.conv_down(x)

        if self.has_se:
            conv2 = self.se(conv2)

        y = self.add(conv2, residual)
        y = self.relu(y)
        return y


class SELayer(nn.Layer):
    def __init__(self, num_channels, num_filters, reduction_ratio, name=None):
        super(SELayer, self).__init__()

        self.pool2d_gap = nn.AdaptiveAvgPool2D(1)  # 图片大小变成1*1，通道数不变

        self._num_channels = num_channels

        med_ch = int(num_channels / reduction_ratio)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self.squeeze = nn.Linear(
            num_channels,
            med_ch,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(-stdv, stdv)))  # 创建一个参数属性对象，设置初始化方式为均匀分布

        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = nn.Linear(
            med_ch,
            num_filters,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(-stdv, stdv)))   # 创建一个参数属性对象，设置初始化方式为均匀分布

    def forward(self, x):
        pool = self.pool2d_gap(x)
        pool = paddle.reshape(pool, shape=[-1, self._num_channels])
        squeeze = self.squeeze(pool)
        squeeze = F.relu(squeeze)
        excitation = self.excitation(squeeze)
        excitation = F.sigmoid(excitation)
        excitation = paddle.reshape(
            excitation, shape=[-1, self._num_channels, 1, 1])
        out = x * excitation
        return out


class Stage(nn.Layer):
    def __init__(self,
                 num_channels,  # [48,96]
                 num_modules,  # 1
                 num_blocks,  # [4,4]
                 num_filters,  # [48,96]
                 has_se=False,
                 multi_scale_output=True,
                 name=None,
                 align_corners=False,
                 padding_same=True):
        super(Stage, self).__init__()

        self._num_modules = num_modules   # 1
        # num_modules表示一个融合块中要进行几次融合，前几次融合是将其他分支的特征融合到最高分辨率的特征图上，只输出最高分辨率特征图（multi_scale_output = False）
        # 只有最后一次的融合是将所有分支的特征融合到每个特征图上，输出所有尺寸特征图（multi_scale_output=True）
        self.stage_func_list = []  # 里面只有一块基础块，因为我们的num-modules=1，所有range中只能取到0
        for i in range(num_modules):  # num_modules=0
            if i == num_modules - 1 and not multi_scale_output:   # and前后都为Ture，则if语句进行
                stage_func = self.add_sublayer(
                    "stage_{}_{}".format(name, i + 1),
                    HighResolutionModule(
                        num_channels=num_channels,
                        num_blocks=num_blocks,
                        num_filters=num_filters,
                        has_se=has_se,
                        multi_scale_output=False,
                        name=name + '_' + str(i + 1),
                        align_corners=align_corners,
                        padding_same=padding_same))
            else:
                stage_func = self.add_sublayer(    # str2执行此模块
                    "stage_{}_{}".format(name, i + 1),
                    HighResolutionModule(
                        num_channels=num_channels,  # [48,96]
                        num_blocks=num_blocks,     # [4,4]
                        num_filters=num_filters,   # [48,96]
                        has_se=has_se,       # False
                        name=name + '_' + str(i + 1),  # str2_1
                        align_corners=align_corners,
                        padding_same=padding_same))

            self.stage_func_list.append(stage_func)

    def forward(self, x):
        out = x
        for idx in range(self._num_modules):
            out = self.stage_func_list[idx](out)
        return out  # [(h/4,w/4,48),(h/8,w/8,96)]


class HighResolutionModule(nn.Layer):
    def __init__(self,
                 num_channels,  # [48,96]
                 num_blocks,   # [4,4]
                 num_filters,    # [48,96]
                 has_se=False,
                 multi_scale_output=True,
                 name=None,
                 align_corners=False,
                 padding_same=True):
        super(HighResolutionModule, self).__init__()

        self.branches_func = Branches(
            num_blocks=num_blocks,  # [4,4]
            in_channels=num_channels,  # [48,96]
            out_channels=num_filters,   # [48,96]
            has_se=has_se,
            name=name,
            padding_same=padding_same)

        self.fuse_func = FuseLayers(
            in_channels=num_filters,  # [48,96]
            out_channels=num_filters,  # [48,96]
            multi_scale_output=multi_scale_output,  # Ture 多尺寸特征输出
            name=name,
            align_corners=align_corners,
            padding_same=padding_same)

    def forward(self, x):
        out = self.branches_func(x)   # [(h/4,w/4,48),(h/8,w/8,96)]
        out = self.fuse_func(out)     # 融合后的 [(h/4,w/4,48),(h/8,w/8,96)]
        return out


class FuseLayers(nn.Layer):   # 特征融合模块
    def __init__(self,
                 in_channels,  # [48，96]
                 out_channels,  # [48,96]
                 multi_scale_output=True,
                 name=None,
                 align_corners=False,
                 padding_same=True):
        super(FuseLayers, self).__init__()

        self._actual_ch = len(in_channels) if multi_scale_output else 1  # 2
        # 如果self.multi_scale_output为Ture，意味着只需要输出最高分辨率特征图，
        # 即只需要将其他尺寸特征图的特征融合入最高分辨率特征图中
        # 但在stage1中，self.multi_scale_output为True，所以range为2
        self._in_channels = in_channels  # [48,96]
        self.align_corners = align_corners   # False

        self.residual_func_list = []
        for i in range(self._actual_ch):  # i=0,1
            # i表示现在要把所有分支的特征（j）融合入第i分支的特征中
            for j in range(len(in_channels)):  # j=0,1
                # 对于j分支进行上采样或者下采样处理，使j分支的通道数以及shape等于i分支
                if j > i:
                    # j > i表示j通道多于i，但shape小于i，需要上采样
                    residual_func = self.add_sublayer(
                        "residual_{}_layer_{}_{}".format(name, i + 1, j + 1),
                        layers.ConvBN(
                            in_channels=in_channels[j],  # j=1时，in_channels=96
                            out_channels=out_channels[i],  # i=0时，out_channels=48
                            kernel_size=1,
                            bias_attr=False))
                    self.residual_func_list.append(residual_func)
                    # j = i表示j与i为同一个分支，不需要做处理
                elif j < i:
                    # 剩余情况则是，j < i，表示j通道少于i，但shape大于i，需要下采样，利用一层或者多层conv2d进行下采样
                    pre_num_filters = in_channels[j]  # 48
                    for k in range(i - j):  # k=0
                        # 这个for k就是实现多层conv2d，而且只有最后一层加激活函数relu
                        if k == i - j - 1:  # Ture
                            residual_func = self.add_sublayer(
                                "residual_{}_layer_{}_{}_{}".format(
                                    name, i + 1, j + 1, k + 1),
                                layers.ConvBN(
                                    in_channels=pre_num_filters,  # 48
                                    out_channels=out_channels[i],   # 96
                                    kernel_size=3,
                                    stride=2,
                                    padding=1 if not padding_same else 'same',
                                    bias_attr=False))
                            pre_num_filters = out_channels[i]
                        else:
                            residual_func = self.add_sublayer(
                                "residual_{}_layer_{}_{}_{}".format(
                                    name, i + 1, j + 1, k + 1),
                                layers.ConvBNReLU(
                                    in_channels=pre_num_filters,
                                    out_channels=out_channels[j],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1 if not padding_same else 'same',
                                    bias_attr=False))
                            pre_num_filters = out_channels[j]
                        self.residual_func_list.append(residual_func)  # 在第一stage中，此列表中仅有两个模块

    def forward(self, x):  # x=[(h/4,w/4,48),(h/8,w/8,96)]
        outs = []
        residual_func_idx = 0
        for i in range(self._actual_ch):  # i=0,1
            residual = x[i]
            residual_shape = paddle.shape(residual)[-2:]   # shape只取高和宽
            for j in range(len(self._in_channels)):  # j=0，1
                if j > i:  # j=1,i=0
                    # j > i表示j通道多于i，但shape小于i，需要上采样
                    y = self.residual_func_list[residual_func_idx](x[j])  # (h/8,w/8,48)通道数减少，从96减少到48
                    residual_func_idx += 1

                    y = F.interpolate(   # 对y上采样，使其形状和x[i]一致，以便融合
                        y,
                        residual_shape,  # （h/4,w/4)
                        mode='bilinear',   # 双线性插值
                        align_corners=self.align_corners) #
                    residual = residual + y  # 融合
                elif j < i:  # j=0,i=1
                    # j < i，表示j通道少于i，但shape大于i，需要下采样，利用一层或者多层conv2d进行下采样
                    y = x[j]   # y=(h/4,w/4,48)
                    for k in range(i - j):  # k=0 这个for k就是实现多层conv2d，而且只有最后一层加激活函数relu
                        y = self.residual_func_list[residual_func_idx](y)  # （h/8,w/8,96)
                        residual_func_idx += 1

                    residual = residual + y  # 融合

            residual = F.relu(residual)
            outs.append(residual)  # [(h/4,w/4,48),(h/8,w/8,96)]

        return outs


@manager.BACKBONES.add_component
def HRNet_W18_Small_V1(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[1],
        stage1_num_channels=[32],
        stage2_num_modules=1,
        stage2_num_blocks=[2, 2],
        stage2_num_channels=[16, 32],
        stage3_num_modules=1,
        stage3_num_blocks=[2, 2, 2],
        stage3_num_channels=[16, 32, 64],
        stage4_num_modules=1,
        stage4_num_blocks=[2, 2, 2, 2],
        stage4_num_channels=[16, 32, 64, 128],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def HRNet_W18_Small_V2(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[2],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[2, 2],
        stage2_num_channels=[18, 36],
        stage3_num_modules=3,
        stage3_num_blocks=[2, 2, 2],
        stage3_num_channels=[18, 36, 72],
        stage4_num_modules=2,
        stage4_num_blocks=[2, 2, 2, 2],
        stage4_num_channels=[18, 36, 72, 144],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def HRNet_W18(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[18, 36],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[18, 36, 72],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[18, 36, 72, 144],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def HRNet_W30(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[30, 60],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[30, 60, 120],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[30, 60, 120, 240],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def HRNet_W32(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[32, 64],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[32, 64, 128],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[32, 64, 128, 256],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def HRNet_W40(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[40, 80],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[40, 80, 160],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[40, 80, 160, 320],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def HRNet_W44(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[44, 88],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[44, 88, 176],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[44, 88, 176, 352],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def HRNet_W48(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[48, 96],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[48, 96, 192],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[48, 96, 192, 384],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def HRNet_W60(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[60, 120],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[60, 120, 240],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[60, 120, 240, 480],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def HRNet_W64(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[64, 128],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[64, 128, 256],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[64, 128, 256, 512],
        **kwargs)
    return model
