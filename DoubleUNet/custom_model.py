import torch
import torch.nn as nn
from doubleunet_pytorch import (
    Conv2D,
    ASPP,
    encoder2,
    decoder1,
    decoder2,
)


class residual_block(nn.Module):
    def __init__(self, in_c, out_c, dilation=[1, 1, 1]):
        super().__init__()

        # dilation 변경 시, shape에 유의하여야함.
        self.c1 = Conv2D(in_c, out_c, dilation=dilation[0])
        self.c2 = Conv2D(out_c, out_c, dilation=dilation[1])
        self.c3 = Conv2D(out_c * 2, out_c, dilation=dilation[2])

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        s = torch.cat([x1, x2], axis=1)
        x3 = self.c3(s)

        return x3


class encoder1(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool_2x2 = nn.MaxPool2d((2, 2))
        self.pool_4x4 = nn.MaxPool2d((4, 4))

        # For dobuleUnet
        # self.c1 = conv_block(3, 64)
        # self.c2 = conv_block(64, 128)
        # self.c3 = conv_block(128, 256)
        # self.c4 = conv_block(256, 512)

        # For ResNet
        self.c1 = residual_block(3, 64, dilation=[1, 1, 1])
        self.c2 = residual_block(64, 128, dilation=[1, 1, 1])
        self.c3 = residual_block(128, 256, dilation=[1, 1, 1])
        self.c4 = residual_block(256, 512, dilation=[1, 1, 1])

    def forward(self, x):
        x0 = x
        x1 = self.c1(x0)
        x2 = self.c2(x1)
        p2 = self.pool_2x2(x2)
        x3 = self.c3(p2)

        p3 = self.pool_2x2(x3)

        x4 = self.c4(p3)

        p4_1 = self.pool_2x2(x4)
        p4_2 = self.pool_4x4(x4)

        return p4_2, [p4_1, p3, p2, x1]


class build_doubleunet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = encoder1()
        self.a1 = ASPP(512, 64)
        self.d1 = decoder1()
        self.y1 = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.e2 = encoder2()
        self.a2 = ASPP(256, 64)
        self.d2 = decoder2()
        self.y2 = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x0 = x
        x, skip1 = self.e1(x)
        x = self.a1(x)
        x = self.d1(x, skip1)
        y1 = self.y1(x)

        input_x = x0 * self.sigmoid(y1)
        x, skip2 = self.e2(input_x)
        x = self.a2(x)
        x = self.d2(x, skip1, skip2)
        y2 = self.y2(x)

        return y1, y2


if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    model = build_doubleunet()
    y1, y2 = model(x)
