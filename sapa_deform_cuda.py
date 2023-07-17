import torch
import torch.nn as nn
import torch.nn.functional as F
from deformsapa import deform_qk, deform_av


class SAPADeform(nn.Module):
    def __init__(self, dim_y, dim_x=None, up_factor=2, num_point=9, groups=4, embedding_dim=32,
                 qkv_bias=True, high_offset=True, norm=False):
        super().__init__()
        dim_x = dim_x if dim_x is not None else dim_y

        self.up_factor = up_factor
        self.num_point = num_point
        self.groups = groups
        self.embedding_dim = embedding_dim
        self.high_offset = high_offset

        self.norm = norm
        if self.norm:
            self.norm_y = nn.GroupNorm(32, dim_y)
            self.norm_x = nn.GroupNorm(32, dim_x)

        self.q = nn.Conv2d(dim_y, groups * embedding_dim, kernel_size=1, bias=qkv_bias)
        self.k = nn.Conv2d(dim_x, groups * embedding_dim, kernel_size=1, bias=qkv_bias)
        if self.high_offset:
            offset_dim = groups * self.num_point * 2 * self.up_factor ** 2
        else:
            offset_dim = groups * self.num_point * 2
        self.offset = nn.Conv2d(dim_x, offset_dim, kernel_size=1, bias=qkv_bias)

        self.apply(self._init_weights)

    def forward(self, y, x):
        if self.norm:
            x_ = self.norm_x(x)
            y = self.norm_y(y)
        else:
            x_ = x

        q = self.q(y)
        k = self.k(x_)

        if self.high_offset:
            offset = F.pixel_shuffle(self.offset(x_), upscale_factor=self.up_factor)
        else:
            offset = F.interpolate(self.offset(x_), scale_factor=self.up_factor)

        return self.attention(q, k, x, offset)

    def attention(self, q, k, x, offset):
        attn = F.softmax(deform_qk(q, k, offset, self.num_point, self.up_factor, self.groups), dim=2)
        return deform_av(attn, x, offset, self.num_point, self.up_factor, self.groups)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


if __name__ == '__main__':
    x = torch.randn(2, 16, 4, 6).to('cuda')
    y = torch.randn(2, 16, 8, 12).to('cuda')
    sapa = SAPADeform(16).to('cuda')
    print(sapa(y, x).shape)
