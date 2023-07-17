/* 
   CUDA extension for SAPA
   by https://github.com/Teleppo
   modified from https://github.com/myownskyW7/CARAFE
*/
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

int QKForwardLauncher(const at::Tensor features,
                        const at::Tensor masks,
                        const int kernel_size,
                        const int scale_factor,
                        const int batch_size,
                        const int channels,
                        const int height,
                        const int width,
                        at::Tensor output);

int QKBackwardLauncher(const at::Tensor top_grad,
                        const at::Tensor features,
                        const at::Tensor masks,
                        const int kernel_size,
                        const int scale_factor,
                        const int batch_size,
                        const int channels,
                        const int height,
                        const int width,
                        at::Tensor bottom_grad,
                        at::Tensor mask_grad);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int qk_forward_cuda(at::Tensor features, at::Tensor masks,
                              int kernel_size, int scale_factor,
                              at::Tensor output)
{
  CHECK_INPUT(features);
  CHECK_INPUT(masks);
  CHECK_INPUT(output);
  at::DeviceGuard guard(features.device());

  int batch_size = masks.size(0);
  int num_channels = masks.size(1);
  int data_height = masks.size(2);
  int data_width = masks.size(3);

  QKForwardLauncher(features, masks, kernel_size,
                            scale_factor, batch_size, num_channels, data_height,
                            data_width, output);

  return 1;
}

int qk_backward_cuda(at::Tensor top_grad, at::Tensor features,
                               at::Tensor masks, int kernel_size,
                               int scale_factor,
                               at::Tensor bottom_grad, at::Tensor mask_grad)
{
  CHECK_INPUT(top_grad);
  CHECK_INPUT(features);
  CHECK_INPUT(masks);
  CHECK_INPUT(bottom_grad);
  CHECK_INPUT(mask_grad);
  at::DeviceGuard guard(top_grad.device());

  int batch_size = masks.size(0);
  int num_channels = masks.size(1);
  int data_height = masks.size(2);
  int data_width = masks.size(3);

  QKBackwardLauncher(top_grad, features, masks, kernel_size,
                             scale_factor, batch_size, num_channels,
                             data_height, data_width, bottom_grad, mask_grad);

  return 1;
}
