import torch
import torch.nn as nn
import torch.nn.functional as F
from sapa import qk, av


class SAPA(nn.Module):
    def __init__(self, dim_y, dim_x=None, up_factor=2, up_kernel_size=5, embedding_dim=32,
                 qkv_bias=True, norm=True):
        super().__init__()
        dim_x = dim_x if dim_x is not None else dim_y

        self.up_factor = up_factor
        self.up_kernel_size = up_kernel_size
        self.embedding_dim = embedding_dim
        self.norm = norm
        if self.norm:
            self.norm_y = nn.GroupNorm(32, dim_y)
            self.norm_x = nn.GroupNorm(32, dim_x)

        self.q = nn.Conv2d(dim_y, embedding_dim, kernel_size=1, bias=qkv_bias)
        self.k = nn.Conv2d(dim_x, embedding_dim, kernel_size=1, bias=qkv_bias)

        self.apply(self._init_weights)

    def forward(self, y, x):
        if self.norm:
            x_ = self.norm_x(x)
            y = self.norm_y(y)
        else:
            x_ = x

        q = self.q(y)
        k = self.k(x_)

        return self.attention(q, k, x)

    def attention(self, q, k, v):
        attn = F.softmax(qk(q, k, self.up_kernel_size, self.up_factor), dim=1)
        return av(attn, v, self.up_kernel_size, self.up_factor)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


if __name__ == '__main__':
    x = torch.randn(2, 64, 4, 6).to('cuda')
    y = torch.randn(2, 64, 8, 12).to('cuda')
    sapa = SAPA(64).to('cuda')
    print(sapa(y, x).shape)
