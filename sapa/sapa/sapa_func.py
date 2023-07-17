# SAPA by https://github.com/Teleppo

import av_ext
import qk_ext
import torch
from torch.autograd import Function


class QKFunction(Function):

    @staticmethod
    def forward(ctx, query, key, kernel_size, scale_factor):
        assert scale_factor >= 1
        assert query.size(0) == key.size(0)
        assert query.size(-1) == key.size(-1) * scale_factor
        assert query.size(-2) == key.size(-2) * scale_factor
        assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1
        ctx.kernel_size = kernel_size
        ctx.scale_factor = scale_factor
        ctx.query_size = query.size()
        ctx.key_size = key.size()

        n, c, h, w = key.size()
        output = key.new_zeros((n, kernel_size ** 2, h * scale_factor, w * scale_factor))
        if key.is_cuda:
            qk_ext.forward(key, query, kernel_size, scale_factor, output)
        else:
            raise NotImplementedError

        if query.requires_grad or key.requires_grad:
            ctx.save_for_backward(query, key)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda

        query, key = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        scale_factor = ctx.scale_factor

        grad_query = torch.zeros_like(query)
        grad_key = torch.zeros_like(key)
        qk_ext.backward(grad_output.contiguous(), key, query,
                        kernel_size, scale_factor,
                        grad_key, grad_query)

        return grad_query, grad_key, None, None, None


qk = QKFunction.apply


class AVFunction(Function):

    @staticmethod
    def forward(ctx, attn, value, kernel_size, scale_factor):
        assert scale_factor >= 1
        assert attn.size(1) == kernel_size * kernel_size
        assert attn.size(-1) == value.size(-1) * scale_factor
        assert attn.size(-2) == value.size(-2) * scale_factor
        assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1
        ctx.kernel_size = kernel_size
        ctx.scale_factor = scale_factor
        ctx.attn_size = attn.size()
        ctx.value_size = value.size()

        n, c, h, w = value.size()
        output = value.new_zeros((n, c, h * scale_factor, w * scale_factor))
        if value.is_cuda:
            av_ext.forward(value, attn, kernel_size, scale_factor, output)
        else:
            raise NotImplementedError

        if attn.requires_grad or value.requires_grad:
            ctx.save_for_backward(attn, value)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda

        attn, value = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        scale_factor = ctx.scale_factor

        grad_attn = torch.zeros_like(attn)
        grad_value = torch.zeros_like(value)
        av_ext.backward(grad_output.contiguous(), value, attn,
                        kernel_size, scale_factor,
                        grad_value, grad_attn)

        return grad_attn, grad_value, None, None, None


av = AVFunction.apply

