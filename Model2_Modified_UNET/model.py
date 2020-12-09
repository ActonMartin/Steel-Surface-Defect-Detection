import torch.nn as nn
import torch
import torch.nn.functional as F

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU(),
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                out_filters, out_filters, kernel_size=(3, 3), dilation=2, padding=2
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters),
        )

    def forward(self, x):
        residual = self.residual(x)
        conv1 = self.conv_1(residual)
        out = self.conv_2(conv1)
        return residual + out


class ResBlock(nn.Module):
    def __init__(
        self,
        in_filters,
        out_filters,
        dropout_rate=0.2,
        kernel_size=(3, 3),
        stride=1,
        pooling=True,
        drop_out=True,
    ):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.residual = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride),
            nn.LeakyReLU(),
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                out_filters, out_filters, kernel_size=(3, 3), dilation=2, padding=2
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(
                out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(out_filters * 3, out_filters, kernel_size=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters),
        )
        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        residual = self.residual(x)
        conv1 = self.conv_1(x)
        conv2 = self.conv_2(conv1)
        conv3 = self.conv_3(conv2)
        concat = torch.cat((conv1, conv2, conv3), dim=1)
        conv4 = self.conv_4(concat)
        conv_out = conv4 + residual

        if self.drop_out:
            out = self.dropout(conv_out)
        else:
            out = conv_out
        if self.pooling:
            out = self.pool(out)
            return out, conv_out
        else:
            return out


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate=0.2, drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.dropout_1 = nn.Dropout2d(p=dropout_rate)
        self.dropout_2 = nn.Dropout2d(p=dropout_rate)
        self.dropout_3 = nn.Dropout2d(p=dropout_rate)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_filters // 4 + 2 * out_filters, out_filters, (3, 3), padding=1
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(out_filters, out_filters, (2, 2), dilation=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(out_filters * 3, out_filters, (1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters),
        )

    def forward(self, x, skip):
        inp = nn.PixelShuffle(2)(x)
        if self.drop_out:
            inp = self.dropout_1(inp)
        inp = torch.cat([inp, skip], dim=1)
        if self.drop_out:
            inp = self.dropout_2(inp)
        upsample1 = self.conv_1(inp)
        upsample2 = self.conv_2(upsample1)
        upsample3 = self.conv_3(upsample2)
        concat = torch.cat([upsample1, upsample2, upsample3], dim=1)
        upsample4 = self.conv_4(concat)

        if self.drop_out:
            upsample4 = self.dropout_3(upsample4)
        return upsample4


class Model(nn.Module):
    def __init__(self, n_classes):
        super(Model, self).__init__()
        self.n_classes = n_classes

        self.downsample_1 = ResContextBlock(3, 32)
        self.downsample_2 = ResContextBlock(32, 32)
        self.downsample_3 = ResContextBlock(32, 32)

        self.resblock_1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resblock_2 = ResBlock(2 * 32, 4 * 32, 0.2, pooling=True)
        self.resblock_3 = ResBlock(4 * 32, 8 * 32, 0.2, pooling=True)
        self.resblock_4 = ResBlock(8 * 32, 8 * 32, 0.2, pooling=True)
        self.resblock_5 = ResBlock(8 * 32, 8 * 32, 0.2, pooling=False)

        self.upsample_1 = UpBlock(8 * 32, 4 * 32, 0.2)
        self.upsample_2 = UpBlock(4 * 32, 4 * 32, 0.2)
        self.upsample_3 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upsample_4 = UpBlock(2 * 32, 32, drop_out=False)

        self.logits = nn.Conv2d(32, self.n_classes, kernel_size=(1, 1))

    def forward(self, x):
        downsample1 = self.downsample_1(x)
        downsample2 = self.downsample_2(downsample1)
        downsample3 = self.downsample_3(downsample2)

        res1, skip1 = self.resblock_1(downsample3)
        res2, skip2 = self.resblock_2(res1)
        res3, skip3 = self.resblock_3(res2)
        res4, skip4 = self.resblock_4(res3)
        res5 = self.resblock_5(res4)

        upsample1 = self.upsample_1(res5, skip4)
        upsample2 = self.upsample_2(upsample1, skip3)
        upsample3 = self.upsample_3(upsample2, skip2)
        upsample4 = self.upsample_4(upsample3, skip1)
        logits = self.logits(upsample4)
        return F.softmax(logits, dim=1)

if __name__ == "__main__":
    device = torch.device("cuda")
    model = Model(5).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)
    inp = torch.randn(16, 3, 64, 512).to(device)
    out = model(inp)
