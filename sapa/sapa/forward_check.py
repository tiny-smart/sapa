import os

import torch
import torch.nn.functional as F
from sapa_func import qk, av
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def qk_pytorch(q, k, kernel_size=5, scale=2):
    B, C, H, W = k.shape
    q = q.view(B, C, H, scale, W, scale)
    k = F.unfold(k, kernel_size=kernel_size, padding=kernel_size // 2).reshape(
            B, C, kernel_size ** 2, H, W)
    return torch.einsum('ijlmno,ijkln->iklmno', q, k).reshape(
            B, kernel_size ** 2, scale * H, scale * W)


def av_pytorch(attn, x, kernel_size=5, scale=2):
    B, C, H, W = x.shape
    attn = attn.view(B, kernel_size ** 2, H, scale, W, scale)
    x = F.unfold(x, kernel_size=kernel_size, padding=kernel_size // 2).view(
        B, C, kernel_size ** 2, H, W)
    return torch.einsum('iklmno,ijkln->ijlmno', attn, x).reshape(
        B, C, H * scale, W * scale)


def forward_check():
    print("forward check...")
    b = 4
    h = 10
    w = 12
    c = 8
    up_kernel_size = 5
    up_factor = 2
    q = torch.randn(b, c, up_factor * h, up_factor * w).to('cuda')
    k = torch.randn(b, c, h, w).to('cuda')
    qk_check = torch.allclose(qk_pytorch(q, k, up_kernel_size, up_factor),
                              qk(q, k, up_kernel_size, up_factor),
                              atol=1e-5)
    print("qk check:", qk_check)

    attn = torch.randn(b, up_kernel_size ** 2, up_factor * h, up_factor * w).to('cuda')
    v = torch.randn(b, c, h, w).to('cuda')
    av_check = torch.allclose(av_pytorch(attn, v, up_kernel_size, up_factor),
                              av(attn, v, up_kernel_size, up_factor),
                              atol=1e-5)
    print("av check:", av_check)


if __name__ == '__main__':
    forward_check()
