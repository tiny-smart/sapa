import os
import torch
from torch.autograd import gradcheck

from deform_func import deform_qk, deform_av
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def grad_check():
    print("grad check...")
    b = 2
    h = 3
    w = 4
    c = 16
    num_point = 5
    up_factor = 2
    groups = 4

    q = torch.rand(b, c, up_factor * h, up_factor * w,
                   requires_grad=True, device='cuda:0').double()
    k = torch.rand(b, c, h, w,
                   requires_grad=True, device='cuda:0').double()
    o = torch.rand(b, groups * num_point * 2, up_factor * h, up_factor * w,
                   requires_grad=True, device='cuda:0').double()

    qk_check = gradcheck(deform_qk, (q, k, o, num_point, up_factor, groups))
    print("qk check:", qk_check)

    a = torch.randn(b, groups, num_point, up_factor * h, up_factor * w,
                    requires_grad=True, device='cuda:0').double()
    v = torch.randn(b, c, h, w,
                    requires_grad=True, device='cuda:0').double()

    av_check = gradcheck(deform_av, (a, v, o, num_point, up_factor, groups))

    print("av check:", av_check)

if __name__ == "__main__":
    grad_check()
