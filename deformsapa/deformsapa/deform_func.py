import torch
from torch.autograd import Function

import dqk, dav


class DeformQKFunction(Function):

    @staticmethod
    def forward(ctx, query, key, offset, num_point, scale_factor, groups):
        assert scale_factor >= 1
        assert query.size(0) == key.size(0)
        assert query.size(1) == key.size(1)
        assert query.size(2) == key.size(2) * scale_factor
        assert query.size(3) == key.size(3) * scale_factor
        assert offset.size(0) == key.size(0)
        assert offset.size(1) == groups * num_point * 2
        assert offset.size(2) == key.size(2) * scale_factor
        assert offset.size(3) == key.size(3) * scale_factor
        assert query.size(1) % groups == 0
        assert num_point >= 1
        ctx.num_point = num_point
        ctx.scale_factor = scale_factor
        ctx.groups = groups
        ctx.query_size = query.size()
        ctx.key_size = key.size()
        ctx.offset_size = offset.size()

        n, c, h, w = query.size()
        output = query.new_zeros((n, groups, num_point, h, w))
        if query.is_cuda:
            dqk.forward(query, key, offset, num_point, scale_factor, groups, output)
        else:
            raise NotImplementedError

        if query.requires_grad or key.requires_grad or offset.requires_grad:
            ctx.save_for_backward(query, key, offset)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda

        query, key, offset = ctx.saved_tensors
        num_point = ctx.num_point
        scale_factor = ctx.scale_factor
        groups = ctx.groups

        grad_query = torch.zeros_like(query)
        grad_key = torch.zeros_like(key)
        grad_offset = torch.zeros_like(offset)
        dqk.backward(grad_output.contiguous(), query, key, offset,
                          num_point, scale_factor, groups,
                          grad_query, grad_key, grad_offset)

        return grad_query, grad_key, grad_offset, None, None, None


deform_qk = DeformQKFunction.apply


class DeformAVFunction(Function):

    @staticmethod
    def forward(ctx, attn, value, offset, num_point, scale_factor, groups):
        assert scale_factor >= 1
        assert attn.size(0) == value.size(0)
        assert attn.size(1) == groups
        assert attn.size(2) == num_point
        assert attn.size(3) == value.size(2) * scale_factor
        assert attn.size(4) == value.size(3) * scale_factor
        assert offset.size(0) == value.size(0)
        assert offset.size(1) == groups * num_point * 2
        assert offset.size(2) == value.size(2) * scale_factor
        assert offset.size(3) == value.size(3) * scale_factor
        assert num_point >= 1
        ctx.num_point = num_point
        ctx.scale_factor = scale_factor
        ctx.groups = groups
        ctx.attn_size = attn.size()
        ctx.value_size = value.size()
        ctx.offset_size = offset.size()

        n, c, h, w = value.size()
        output = value.new_zeros((n, c, scale_factor * h, scale_factor * w))
        if attn.is_cuda:
            dav.forward(attn, value, offset, num_point, scale_factor, groups, output)
        else:
            raise NotImplementedError

        if attn.requires_grad or value.requires_grad or offset.requires_grad:
            ctx.save_for_backward(attn, value, offset)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda

        attn, value, offset = ctx.saved_tensors
        num_point = ctx.num_point
        scale_factor = ctx.scale_factor
        groups = ctx.groups

        grad_attn = torch.zeros_like(attn)
        grad_value = torch.zeros_like(value)
        grad_offset = torch.zeros_like(offset)
        dav.backward(grad_output.contiguous(), attn, value, offset,
                           num_point, scale_factor, groups,
                           grad_attn, grad_value, grad_offset)

        return grad_attn, grad_value, grad_offset, None, None, None


deform_av = DeformAVFunction.apply
