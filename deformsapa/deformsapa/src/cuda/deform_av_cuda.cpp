/*
   CUDA extension for DeformLCA
   by https://github.com/Poppuppy
*/
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

int DeformAVForwardLauncher(const at::Tensor attn,
						    const at::Tensor value,
						    const at::Tensor offset,
                            const int num_point,
                            const int scale_factor,
                            const int groups,
						    const int batch_size,
                            const int channels,
						    const int height,
                            const int width,
						    at::Tensor output);

int DeformAVBackwardLauncher(const at::Tensor top_grad,
                             const at::Tensor data_attn,
                             const at::Tensor data_value,
                             const at::Tensor data_offset,
						     const int num_point,
						     const int scale_factor,
						     const int groups,
						     const int batch_size,
						     const int channels,
						     const int height,
						     const int width,
						     at::Tensor grad_attn,
						     at::Tensor grad_value,
						     at::Tensor grad_offset);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int deform_av_forward_cuda(at::Tensor attn,
						   at::Tensor value,
						   at::Tensor offset,
                           int num_point,
                           int scale_factor,
                           int groups,
						   at::Tensor output)
{
  CHECK_INPUT(attn);
  CHECK_INPUT(value);
  CHECK_INPUT(offset);
  CHECK_INPUT(output);
  at::DeviceGuard guard(attn.device());

  int batch_size = attn.size(0);
  int num_channels = value.size(1);
  int data_height = attn.size(3);
  int data_width = attn.size(4);

  DeformAVForwardLauncher(attn, value, offset, num_point, scale_factor, groups,
                          batch_size, num_channels, data_height, data_width, output);

  return 1;
}

int deform_av_backward_cuda(at::Tensor top_grad,
                            at::Tensor data_attn,
                            at::Tensor data_value,
                            at::Tensor data_offset,
						    int num_point,
						    int scale_factor,
						    int groups,
						    at::Tensor grad_attn,
						    at::Tensor grad_value,
						    at::Tensor grad_offset)
{
  CHECK_INPUT(top_grad);
  CHECK_INPUT(data_attn);
  CHECK_INPUT(data_value);
  CHECK_INPUT(data_offset);
  CHECK_INPUT(grad_attn);
  CHECK_INPUT(grad_value);
  CHECK_INPUT(grad_offset);
  at::DeviceGuard guard(top_grad.device());

  int batch_size = data_attn.size(0);
  int num_channels = data_value.size(1);
  int data_height = data_attn.size(3);
  int data_width = data_attn.size(4);

  DeformAVBackwardLauncher(top_grad, data_attn, data_value, data_offset, num_point,
                           scale_factor, groups, batch_size, num_channels, data_height, data_width,
						   grad_attn, grad_value, grad_offset);

  return 1;
}
