import torch
from torch import nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class RepConvN(nn.Module):
    # Basic Rep-style Convolutional Block
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))



class MSFAA(nn.Module):
    def __init__(self, channels, factor=16):
        super(MSFAA, self).__init__()
        self.groups = factor
        assert channels % self.groups == 0, "Channels must be divisible by groups"
        self.softmax = nn.Softmax(-1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.criss_cross_attention = nn.Conv2d(channels // self.groups * 2, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.squeeze_and_excitation = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        b, c, h, w = x.size()
        group_channels = c // self.groups
        group_x = x.reshape(b * self.groups, group_channels, h, w)

        x_avg = self.global_avg_pool(group_x)
        x_max = self.global_max_pool(group_x)

        combined = torch.cat([x_avg, x_max], dim=1)  # Concatenate along channels
        attention_weights = torch.sigmoid(self.criss_cross_attention(combined))
        x_attention = group_x * attention_weights

        se_weights = self.squeeze_and_excitation(x_attention)
        x1 = x_attention + se_weights

        x2 = self.conv3x3(group_x) + self.conv5x5(group_x)

        x11 = self.softmax(self.global_avg_pool(x1).view(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.view(b * self.groups, group_channels, -1)
        x21 = self.softmax(self.global_avg_pool(x2).view(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.view(b * self.groups, group_channels, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).view(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).view(b, c, h, w)



class RepNCSPELAN4_MSFAA(nn.Module):
    # CSP-ELAN with new EMA structure
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, c5), MSFAA(c4), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), MSFAA(c4), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


# Additional Helper Classes
class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))



