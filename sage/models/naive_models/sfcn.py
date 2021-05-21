import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class SFCN(nn.Module):
    def __init__(self, cfg):
        super(SFCN, self).__init__()
        channel_number = cfg.channel_number if cfg else [32, 64, 128, 256, 256, 64]
        dropout = cfg.dropout
        output_dim = 1
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        self.classifier = nn.Sequential()
        avg_shape = [2, 2, 2]
        # avg_shape = [5, 5, 5]
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        out = list()
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        # x = F.log_softmax(x, dim=1)
        # out.append(x)
        # return out
        return x

if __name__=="__main__":

    class CFG:
        channel_number = [32, 64, 128, 256, 256, 64]
        output_dim = 1
        dropout = True
    cfg = CFG()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SFCN(cfg=cfg).to(device)
    print(model)
    print(summary(model, input_size=(1, 96, 96, 96)))

    # sample = torch.zeros((2, 1, 96, 96, 96)).to(device)
    # print(model(sample).squeeze(1))
