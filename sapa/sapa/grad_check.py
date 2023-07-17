import os
import torch
from torch.autograd import gradcheck

from sapa_func import qk, av


def grad_check():
    print("grad check...")
    b = 2
    h = 3
    w = 4
    c = 16
    up_kernel_size = 5
    up_factor = 2

    q = torch.rand(b, c, up_factor * h, up_factor * w,
                   requires_grad=True, device='cuda:0').double()
    k = torch.rand(b, c, h, w,
                   requires_grad=True, device='cuda:0').double()

    qk_check = gradcheck(qk, (q, k, up_kernel_size, up_factor))
    print("qk check:", qk_check)

    a = torch.randn(b, up_kernel_size ** 2, up_factor * h, up_factor * w,
                    requires_grad=True, device='cuda:0').double()
    v = torch.randn(b, c, h, w,
                    requires_grad=True, device='cuda:0').double()

    av_check = gradcheck(av, (a, v, up_kernel_size, up_factor), eps=1e-4)
    # av_check = gradcheck(av, (v, a, up_kernel_size, 1, up_factor), eps=1e-4)

    print("av check:", av_check)


if __name__ == "__main__":
    grad_check()
