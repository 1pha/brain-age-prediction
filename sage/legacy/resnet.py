import math
from functools import partial

# from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def get_inplanes(start_channels=16):
    if start_channels == 8:
        return [8, 16, 32, 64]

    elif start_channels == 16:
        return [16, 32, 64, 128]

    elif start_channels == 32:
        return [32, 64, 128, 256]

    elif start_channels == 64:
        return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, activation=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True) if activation is None else activation
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        n_input_channels=3,
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        shortcut_type="B",
        widen_factor=1.0,
        n_classes=400,
        activation=None,
    ):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True) if activation is None else activation
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, block_inplanes[0], layers[0], shortcut_type, activation=activation
        )
        self.layer2 = self._make_layer(
            block,
            block_inplanes[1],
            layers[1],
            shortcut_type,
            stride=2,
            activation=activation,
        )
        self.layer3 = self._make_layer(
            block,
            block_inplanes[2],
            layers[2],
            shortcut_type,
            stride=2,
            activation=activation,
        )
        self.layer4 = self._make_layer(
            block,
            block_inplanes[3],
            layers[3],
            shortcut_type,
            stride=2,
            activation=activation,
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(
            out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
        )
        # if isinstance(out.data, torch.cuda.FloatTensor):
        #     zero_pads = zero_pads.cuda()
        zero_pads = zero_pads.to(out.device)
        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(
        self, block, planes, blocks, shortcut_type, stride=1, activation=None
    ):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                activation=activation,
            )
        )
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        return x

    # @property
    def conv_layers(self):

        conv_layers = [self.conv1]
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for l in layer:
                conv_layers.append(l.conv1)
                conv_layers.append(l.conv2)

        return conv_layers


def generate_model(model_depth, start_channels, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(start_channels), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(start_channels), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


class Option:
    def __init__(
        self,
        model_depth=10,
        n_classes=1,
        n_input_channels=1,
        shortcut_type="A",
        conv1_t_size=7,
        conv1_t_stride=2,
        no_max_pool=False,
        start_channels=32,
    ):

        self.model_depth = model_depth
        self.n_classes = n_classes
        self.n_input_channels = n_input_channels
        self.shortcut_type = shortcut_type
        self.conv1_t_size = conv1_t_size
        self.conv1_t_stride = conv1_t_stride
        self.no_max_pool = no_max_pool
        self.start_channels = start_channels


def load_resnet(*args, **kwargs):

    opt = Option(**kwargs)
    opt = vars(opt)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return generate_model(**opt).to(device)


if __name__ == "__main__":

    opt = Option()
    opt.start_channels = 32
    opt = edict(vars(opt))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = generate_model(**opt)
    model.to(device)

    # print(summary(model, input_size=(1, 96, 96, 96)))
    sample = torch.zeros(2, 1, 96, 96, 96, device=device)
    print(model.forward(sample).shape)


import torch.nn as nn

class NKRegressor(nn.Module):

    def __init__(self, cfg=None):
        super(NKRegressor, self).__init__()

        if cfg is None:
            init_node = 256
        else:
            self.cfg = cfg
            init_node = self.cfg.init_node

        self.regressor = nn.Sequential(
            nn.Linear(init_node, init_node // 2),
            nn.ReLU(),
            nn.Linear(init_node // 2, init_node // 4),
            nn.ReLU(),
            nn.Linear(init_node // 4, 1)
        )

    def forward(self, x):
        out = self.regressor(x)
        return out


class NKDomainPredictor(nn.Module):
    
    def __init__(self, cfg=None):
        super(NKDomainPredictor, self).__init__()

        if cfg is None:
            num_dbs = 2
        else:
            self.cfg = cfg
            init_node = self.cfg.init_node
            num_dbs = self.cfg.num_dbs

        self.domain = nn.Sequential(
            nn.Linear(init_node, init_node // 2),
            nn.ReLU(),
            nn.Dropout3d(p=.2),
            nn.Linear(init_node // 2, init_node // 4),
            nn.ReLU(),
            nn.Linear(init_node // 4, num_dbs),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.domain(x)
        return out


load_predictors = {
    'nkregressor': NKRegressor,
    'nkdomainpredictor': NKDomainPredictor,
}


class Assembled(nn.Module):
    def __init__(self, encoder, regressor):

        super().__init__()
        self.encoder = encoder
        self.regressor = regressor

    def load_weight(self, weights: dict):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        for model_name, path in weights.items():

            if model_name == "encoder":
                self.encoder.load_state_dict(torch.load(path, map_location=device))

            elif model_name == "regressor":
                self.regressor.load_state_dict(torch.load(path, map_location=device))

        print("Weights successfully loaded!")

    def forward(self, x):

        out = self.encoder(x)
        out = self.regressor(out)

        return out

    @property
    def conv_layers(self):

        try:
            return self.encoder.conv_layers

        except:
            print("No conv_layers attribute supported for this model !")
            return