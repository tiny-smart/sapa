/*
   CUDA extension for DeformLCA
   by https://github.com/Poppuppy
*/
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

int DeformQKForwardLauncher(const at::Tensor query,
						    const at::Tensor key,
						    const at::Tensor offset,
                            const int num_point,
                            const int scale_factor,
                            const int groups,
						    const int batch_size,
                            const int channels,
						    const int height,
                            const int width,
						    at::Tensor output);

int DeformQKBackwardLauncher(const at::Tensor top_grad,
                             const at::Tensor data_query,
                             const at::Tensor data_key,
                             const at::Tensor data_offset,
						     const int num_point,
						     const int scale_factor,
						     const int groups,
						     const int batch_size,
						     const int channels,
						     const int height,
						     const int width,
						     at::Tensor grad_query,
						     at::Tensor grad_key,
						     at::Tensor grad_offset);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int deform_qk_forward_cuda(at::Tensor query,
						   at::Tensor key,
						   at::Tensor offset,
                           int num_point,
                           int scale_factor,
                           int groups,
						   at::Tensor output)
{
  CHECK_INPUT(query);
  CHECK_INPUT(key);
  CHECK_INPUT(offset);
  CHECK_INPUT(output);
  at::DeviceGuard guard(query.device());

  int batch_size = query.size(0);
  int num_channels = query.size(1);
  int data_height = query.size(2);
  int data_width = query.size(3);

  DeformQKForwardLauncher(query, key, offset, num_point, scale_factor, groups,
                          batch_size, num_channels, data_height, data_width, output);

  return 1;
}

int deform_qk_backward_cuda(at::Tensor top_grad,
                            at::Tensor data_query,
                            at::Tensor data_key,
                            at::Tensor data_offset,
						    int num_point,
						    int scale_factor,
						    int groups,
						    at::Tensor grad_query,
						    at::Tensor grad_key,
						    at::Tensor grad_offset)
{
  CHECK_INPUT(top_grad);
  CHECK_INPUT(data_query);
  CHECK_INPUT(data_key);
  CHECK_INPUT(data_offset);
  CHECK_INPUT(grad_query);
  CHECK_INPUT(grad_key);
  CHECK_INPUT(grad_offset);
  at::DeviceGuard guard(top_grad.device());

  int batch_size = data_query.size(0);
  int num_channels = data_query.size(1);
  int data_height = data_query.size(2);
  int data_width = data_query.size(3);

  DeformQKBackwardLauncher(top_grad, data_query, data_key, data_offset, num_point,
                           scale_factor, groups, batch_size, num_channels, data_height, data_width,
						   grad_query, grad_key, grad_offset);

  return 1;
}
