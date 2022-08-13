import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer.activation import GLU
from conformer.encoder import ConformerBlock


class EncoderLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(EncoderLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, ipt):
        return self.main(ipt)


class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            GLU(dim=1),
        )

    def forward(self, ipt):
        return self.main(ipt)


class Middle(nn.Module):
    def __init__(self, encoder_dim, num_attention_heads):
        super(Middle, self).__init__()
        self.main = nn.Sequential(ConformerBlock(encoder_dim=encoder_dim, num_attention_heads=num_attention_heads))

    def forward(self, ipt):
        return self.main(ipt)


class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            GLU(dim=1),
        )

    def forward(self, ipt):
        return self.main(ipt)


class DecoderInLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DecoderInLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, ipt):
        return self.main(ipt)

class DecoderOutLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=1, stride=1):
        super(DecoderOutLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size, stride=stride),
            nn.Tanh()
        )

    def forward(self, ipt):
        return self.main(ipt)


class FTSC_GAN(nn.Module):
    def __init__(self, initial_channel=12, n_layers=12, channels_interval=24, conformer_nums=2):
        super(FTSC_GAN, self).__init__()

        self.n_layers = n_layers
        self.conformer_nums = conformer_nums
        self.channels_interval = channels_interval

        self.encoder_layer = EncoderLayer(1, initial_channel)

        downsample_in_channels_list = [initial_channel] + [i * self.channels_interval for i in range(1, self.n_layers)]
        downsample_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]

        self.down_sample = nn.ModuleList()
        for i in range(self.n_layers):
            self.down_sample.append(
                DownSamplingLayer(
                    channel_in=downsample_in_channels_list[i],
                    channel_out=downsample_out_channels_list[i] * 2
                )
            )

        self.middle = Middle(encoder_dim=self.n_layers * self.channels_interval, num_attention_heads=4)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


        upsample_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        upsample_in_channels_list = upsample_in_channels_list[::-1]
        upsample_out_channels_list = downsample_out_channels_list[::-1]

        self.upsmaple = nn.ModuleList()

        for i in range(self.n_layers):
            self.upsmaple.append(
                UpSamplingLayer(
                    channel_in=upsample_in_channels_list[i],
                    channel_out=upsample_out_channels_list[i] * 2
                )
            )
        self.decoder_in_layer = DecoderInLayer(initial_channel + channels_interval, 1, stride=2)
        self.decoder_out_layer = DecoderOutLayer(2, 1)

    def forward(self, input):
        tmp = []
        o = input
        o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)

        encoder_out = self.encoder_layer(o)

        o = encoder_out
        for i in range(self.n_layers):
            o = self.down_sample[i](o)
            tmp.append(o)
            o = o[:, :, ::2]

        o = o.permute(0, 2, 1)

        for i in range(self.conformer_nums):
            co = self.middle(o)
            o = self.relu(co + o)
        o = o.permute(0, 2, 1)


        for i in range(self.n_layers):

            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            o = torch.cat([o, tmp[self.n_layers - i - 1]], dim=1)
            o = self.upsmaple[i](o)

        o = torch.cat([o, encoder_out], dim=1)
        o = self.decoder_in_layer(o)
        o = torch.cat([o, input], dim=1)
        o = self.decoder_out_layer(o)

        return o


if __name__ == '__main__':
    model = FTSC_GAN(initial_channel=12)
    inputs = torch.randn(8, 1, 16384)
    print(model)
    print(model(inputs).shape)