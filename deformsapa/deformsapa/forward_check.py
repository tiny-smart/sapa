import torch
import torch.nn.functional as F
from deform_func import deform_qk, deform_av


def sample(x, offset, num_point=5, scale=2, groups=4):
    B, _, H, W = x.shape
    offset = offset.view(B, -1, 2, scale * H, scale * W).permute(
        0, 1, 3, 4, 2).contiguous()
    coords_h = torch.arange(H) + 0.5
    coords_w = torch.arange(W) + 0.5
    coords = torch.stack(torch.meshgrid([coords_w, coords_h])).permute(2, 1, 0).type(offset.dtype).to(offset.device)
    coords = coords.view(H, 1, W, 1, 2).repeat(1, scale, 1, scale, 1).view(scale * H, scale * W, 2)
    normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 1, 1, 1, 2)
    coords = 2 * (coords.unsqueeze(0).unsqueeze(0) + offset) / normalizer - 1
    return F.grid_sample(x.view(B * groups, -1, H, W).repeat(1, num_point, 1, 1).view(
        B * groups * num_point, -1, H, W),
        coords.contiguous().view(-1, scale * H, scale * W, 2), mode='bilinear',
        align_corners=False).view(B, groups, num_point, -1, scale * H, scale * W)


def deform_qk_pytorch(q, k, offset, num_point=5, up_factor=2, groups=4):
    # q = q.permute(0, 3, 1, 2).contiguous()
    # k = k.permute(0, 3, 1, 2).contiguous()
    B, _, H, W = q.shape
    q = q.view(B, groups, 1, -1, H, W)
    k = sample(k, offset, num_point=num_point, scale=up_factor, groups=groups)
    attn = torch.sum(q * k, dim=3)
    return attn


def deform_av_pytorch(attn, v, offset, num_point=5, up_factor=2, groups=4):
    # attn = attn.permute(0, 3, 1, 2).contiguous()
    # v = v.permute(0, 3, 1, 2).contiguous()
    # B, H, W, _ = v.shape
    B, _, _, H, W = attn.shape
    attn = attn.unsqueeze(3)
    v = sample(v, offset, num_point=num_point, scale=up_factor, groups=groups)
    v = torch.sum(attn * v, dim=2).view(B, -1, H, W)
    return v


def forward_check():
    print("forward check...")
    b = 4
    h = 10
    w = 12
    c = 8
    num_point = 5
    up_factor = 2
    groups = 4
    q = torch.randn(b, c, up_factor * h, up_factor * w).to('cuda')
    k = torch.randn(b, c, h, w).to('cuda')
    offset = torch.randn(b, groups * num_point * 2, up_factor * h, up_factor * w).to('cuda')
    qk_check = torch.allclose(deform_qk_pytorch(q, k, offset, num_point, up_factor, groups),
                              deform_qk(q, k, offset, num_point, up_factor, groups),
                              atol=1e-5)
    print("qk check:", qk_check)

    attn = torch.randn(b, groups, num_point, up_factor * h, up_factor * w).to('cuda')
    v = torch.randn(b, c, h, w).to('cuda')
    av_check = torch.allclose(deform_av_pytorch(attn, v, offset, num_point, up_factor, groups),
                              deform_av(attn, v, offset, num_point, up_factor, groups),
                              atol=1e-5)
    print("av check:", av_check)


if __name__ == '__main__':
    forward_check()
