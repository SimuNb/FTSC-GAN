import torch
from torch.autograd import Variable
import torch.nn as nn
from conformer.modules import Transpose
negative_slope = 0.03
class Discriminator_T(nn.Module):
    def __init__(self):
        super(Discriminator_T, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=31, stride=2, padding=15),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=31, stride=2, padding=15),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=31, stride=2, padding=15),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=31, stride=2, padding=15),
            nn.LeakyReLU(negative_slope),
            Transpose(shape=(1, 2)),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def normal_z(self, in_tensor):
        z = nn.init.normal_(torch.Tensor(in_tensor.shape))
        if in_tensor.is_cuda:
            z = z.cuda()
        z = Variable(z)
        return z

    def forward(self, inp):
        inp = torch.cat([inp, self.normal_z(inp)], dim=1)
        return self.sequential(inp).view(-1)

def main():
    model = Discriminator_T()
    a = torch.randn(8, 2, 16384)
    b = model(a)
    print(b.shape)
if __name__ == "__main__":
    main()