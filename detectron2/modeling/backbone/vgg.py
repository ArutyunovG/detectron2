import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import ShapeSpec

from .backbone import Backbone
from .build import BACKBONE_REGISTRY


class SSD_VGG(Backbone):
    def __init__(self, input_shape):

        super(SSD_VGG, self).__init__()

        self.conv1_1 = nn.Conv2d(
            in_channels=input_shape.channels,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
            padding=1,
        )
        self.conv1_2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
            padding=1,
        )
        self.conv2_1 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
            padding=1,
        )
        self.conv2_2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
            padding=1,
        )
        self.conv3_1 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
            padding=1,
        )
        self.conv3_2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
            padding=1,
        )
        self.conv3_3 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
            padding=1,
        )
        self.conv4_1 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
            padding=1,
        )
        self.conv4_2 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
            padding=1,
        )
        self.conv4_3 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
            padding=1,
        )
        self.conv5_1 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
            padding=1,
        )
        self.conv5_2 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
            padding=1,
        )
        self.conv5_3 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
            padding=1,
        )
        self.fc6 = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
            dilation=6,
            padding=6,
        )
        self.fc7 = nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv6_1 = nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv6_2 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(2, 2),
            groups=1,
            bias=True,
            padding=1,
        )
        self.conv7_1 = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv7_2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(2, 2),
            groups=1,
            bias=True,
            padding=1,
        )
        self.conv8_1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv8_2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
            padding=1,
        )
        self.conv9_1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv9_2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=True,
            padding=1,
        )

        self.conv4_3_norm_param = nn.parameter.Parameter(torch.Tensor(1, 512, 1, 1))

    def forward(self, x):

        conv1_1 = self.conv1_1(x)
        relu1_1 = F.relu(conv1_1)
        conv1_2 = self.conv1_2(relu1_1)
        relu1_2 = F.relu(conv1_2)
        pool1 = F.max_pool2d(relu1_2, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=True)
        conv2_1 = self.conv2_1(pool1)
        relu2_1 = F.relu(conv2_1)
        conv2_2 = self.conv2_2(relu2_1)
        relu2_2 = F.relu(conv2_2)
        pool2 = F.max_pool2d(relu2_2, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=True)
        conv3_1 = self.conv3_1(pool2)
        relu3_1 = F.relu(conv3_1)
        conv3_2 = self.conv3_2(relu3_1)
        relu3_2 = F.relu(conv3_2)
        conv3_3 = self.conv3_3(relu3_2)
        relu3_3 = F.relu(conv3_3)
        pool3 = F.max_pool2d(relu3_3, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=True)
        conv4_1 = self.conv4_1(pool3)
        relu4_1 = F.relu(conv4_1)
        conv4_2 = self.conv4_2(relu4_1)
        relu4_2 = F.relu(conv4_2)
        conv4_3 = self.conv4_3(relu4_2)
        relu4_3 = F.relu(conv4_3)
        pool4 = F.max_pool2d(relu4_3, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=True)
        conv5_1 = self.conv5_1(pool4)
        relu5_1 = F.relu(conv5_1)
        conv5_2 = self.conv5_2(relu5_1)
        relu5_2 = F.relu(conv5_2)
        conv5_3 = self.conv5_3(relu5_2)
        relu5_3 = F.relu(conv5_3)
        pool5 = F.max_pool2d(relu5_3, kernel_size=(3, 3), stride=(1, 1), padding=1, ceil_mode=True)
        fc6 = self.fc6(pool5)
        relu6 = F.relu(fc6)
        fc7 = self.fc7(relu6)
        relu7 = F.relu(fc7)
        conv6_1 = self.conv6_1(relu7)
        conv6_1_relu = F.relu(conv6_1)
        conv6_2 = self.conv6_2(conv6_1_relu)
        conv6_2_relu = F.relu(conv6_2)
        conv7_1 = self.conv7_1(conv6_2_relu)
        conv7_1_relu = F.relu(conv7_1)
        conv7_2 = self.conv7_2(conv7_1_relu)
        conv7_2_relu = F.relu(conv7_2)
        conv8_1 = self.conv8_1(conv7_2_relu)
        conv8_1_relu = F.relu(conv8_1)
        conv8_2 = self.conv8_2(conv8_1_relu)
        conv8_2_relu = F.relu(conv8_2)
        conv9_1 = self.conv9_1(conv8_2_relu)
        conv9_1_relu = F.relu(conv9_1)
        conv9_2 = self.conv9_2(conv9_1_relu)
        conv9_2_relu = F.relu(conv9_2)

        conv4_3_norm = F.normalize(relu4_3, p=2, dim=1)
        conv4_3_norm = conv4_3_norm * self.conv4_3_norm_param

        return {
            "conv4_3_norm": conv4_3_norm,
            "relu7": relu7,
            "conv6_2_relu": conv6_2_relu,
            "conv7_2_relu": conv7_2_relu,
            "conv8_2_relu": conv8_2_relu,
            "conv9_2_relu": conv9_2_relu,
        }

    def output_shape(self):
        return {
            "conv4_3_norm": ShapeSpec(channels=512, stride=8),
            "relu7": ShapeSpec(channels=1024, stride=16),
            "conv6_2_relu": ShapeSpec(channels=512, stride=32),
            "conv7_2_relu": ShapeSpec(channels=256, stride=64),
            "conv8_2_relu": ShapeSpec(channels=256, stride=100),
            "conv9_2_relu": ShapeSpec(channels=256, stride=300),
        }


@BACKBONE_REGISTRY.register()
def build_ssd_vgg_backbone(cfg, input_shape: ShapeSpec):
    return SSD_VGG(input_shape)
