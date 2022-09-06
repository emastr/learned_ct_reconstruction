import torch
import torch.nn as nn
from odl.contrib.torch import OperatorModule as OperatorModule  # OperatorModule 3.6

# SET SEED
seed = 123
torch.manual_seed(seed)
torch.set_default_dtype(torch.double)


class OpCorrection(nn.Module):

    def __init__(self, operator, **kwargs):
        super(OpCorrection, self).__init__()
        in_channels = operator.range.shape[0]
        self.opModule = OperatorModule(operator)
        self.unet = Unet(in_channels=in_channels, **kwargs)

    def forward(self, x):
        op_x = self.opModule(x)
        return self.unet(op_x) + op_x


class Unet(nn.Module):
    """
    U-net as presented in the article by Ronneberger,
    But with padding to conserve image dimension.
    """
    def __init__(self, in_channels, min_out_channels, depth, padding_mode="circular", batch_norm=0, activation=nn.ReLU):
        super(Unet, self).__init__()
        self.expansion = Expansion(min_out_channels, depth, padding_mode=padding_mode, batch_norm=batch_norm, activation=activation)
        self.contraction = Contraction(in_channels, min_out_channels, depth, padding_mode=padding_mode, batch_norm=batch_norm, activation=activation)
        self.segmentation = nn.Conv2d(in_channels=min_out_channels, out_channels=in_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.segmentation(self.expansion(self.contraction(x)))


class ResUnet(nn.Module):
    def __init__(self, in_channels, min_out_channels, depth, scale, padding_mode="circular", batch_norm=0, activation=nn.ReLU):
        """
        Scale should be small for better initialisation.
        :param in_channels:
        :param min_out_channels:
        :param depth:
        :param scale:
        """
        super(ResUnet, self).__init__()
        self.scale = scale
        self.unet = Unet(in_channels=in_channels, min_out_channels=min_out_channels, depth=depth, padding_mode=padding_mode, batch_norm=batch_norm, activation=activation)
        if scale is not None:
            self.unet.apply(self.weight_init)

    def weight_init(self, module: nn.Module):
        # Class name
        classname = module.__class__.__name__
        if (classname == 'Conv2d') or (classname == 'ConvTranspose2d'):
            module.weight.data.normal_(0, std=self.scale)
            module.bias.data.normal_(0, std=self.scale)

    def forward(self, x):
        return self.unet(x) + x


class Contraction(nn.Module):
    def __init__(self, in_channels, min_out_channels, depth, padding_mode="circular", batch_norm=0, activation=nn.ReLU):
        super(Contraction, self).__init__()
        self.convBlocks = nn.ModuleList([])
        self.maxPools = nn.ModuleList([])
        self.depth = depth

        out_channels = min_out_channels
        for d in range(depth):
            self.convBlocks.append(ConvBlock(in_channels, out_channels, padding_mode=padding_mode, batch_norm=batch_norm, activation=activation))
            if d < depth:
                self.maxPools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            out_channels = out_channels * 2


    def forward(self, x):
        outputs: list = [self.convBlocks[0](x)]
        for d in range(1, self.depth):
            outputs.append(self.convBlocks[d](self.maxPools[d-1](outputs[-1])))
        return outputs


class Expansion(nn.Module):
    def __init__(self, min_out_channels, depth, padding_mode="circular", batch_norm=0, activation=nn.ReLU):
        super(Expansion, self).__init__()
        self.convBlocks = nn.ModuleList([])
        self.upConvs = nn.ModuleList([])
        self.depth = depth

        out_channels = min_out_channels
        for d in range(depth-1):
            self.convBlocks.append(ConvBlock(in_channels=2 * out_channels, out_channels=out_channels, padding_mode=padding_mode, batch_norm=batch_norm, activation=activation))
            self.upConvs.append(nn.ConvTranspose2d(in_channels=2 * out_channels, out_channels=out_channels, kernel_size=2, stride=2))
            out_channels = out_channels * 2

    def forward(self, x: list):
        out = x[-1]
        for d in reversed(range(self.depth - 1)):
            out = self.convBlocks[d](torch.cat([x[d], self.upConvs[d](out)], dim=1))
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode="circular", batch_norm=0, activation=nn.ReLU):
        """
        Convolution block (C_i, X, Y) -> conv2d
                       -> (C_o, X-2, Y-2) -> reLU
                       -> (C_o, X-2, Y-2) -> conv2d
                       -> (C_o, X-4, Y-4) -> reLu
                       -> (C_o, X-4, Y-4) -> interpolate
                       -> (C_o, X, Y)

        :param in_channels: Number of channels in input image
        :param out_channels: Number of features in output
        """

        super(ConvBlock, self).__init__()
        if batch_norm == 0:
            self.sequential = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=3,
                                                      padding=1,
                                                      padding_mode=padding_mode),
                                            activation(),
                                            #nn.ReLU(),
                                            nn.Conv2d(in_channels=out_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=3,
                                                      padding=1,
                                                      padding_mode=padding_mode),
                                            activation())
                                            #nn.ReLU())
        elif batch_norm == 1:
            self.sequential = nn.Sequential(nn.utils.weight_norm(nn.Conv2d(in_channels=in_channels,
                                                              out_channels=out_channels,
                                                              kernel_size=3,
                                                              padding=1,
                                                              padding_mode=padding_mode)),
                                            #nn.ReLU(),
                                            activation(),
                                            nn.utils.weight_norm(nn.Conv2d(in_channels=out_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=3,
                                                      padding=1,
                                                      padding_mode=padding_mode)),
                                            #nn.ReLU())
                                            activation())
        elif batch_norm == 2:
            self.sequential = nn.Sequential(nn.BatchNorm2d(in_channels),
                                            nn.Conv2d(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=3,
                                                      padding=1,
                                                      padding_mode=padding_mode),
                                            # nn.ReLU(),
                                            nn.BatchNorm2d(out_channels),
                                            activation(),
                                            nn.Conv2d(in_channels=out_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=3,
                                                      padding=1,
                                                      padding_mode=padding_mode),
                                            # nn.ReLU())
                                            activation())
        else:
            assert False, f"Illegal argument for batch_norm: {batch_norm}"


    def forward(self, x):
        return self.sequential(x)

