#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

int AVForwardLauncher(const at::Tensor features, const at::Tensor masks,
                              const int kernel_size,
                              const int scale_factor, const int batch_size,
                              const int channels, const int height,
                              const int width, at::Tensor output);

int AVBackwardLauncher(const at::Tensor top_grad,
                               const at::Tensor features,
                               const at::Tensor masks, const int kernel_size,
                               const int scale_factor,
                               const int batch_size, const int channels,
                               const int height, const int width,
                               at::Tensor bottom_grad, at::Tensor mask_grad);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int av_forward_cuda(at::Tensor features, at::Tensor masks,
                              int kernel_size, int scale_factor,
                              at::Tensor output)
{
  CHECK_INPUT(features);
  CHECK_INPUT(masks);
  CHECK_INPUT(output);
  at::DeviceGuard guard(features.device());

  int batch_size = output.size(0);
  int num_channels = output.size(1);
  int data_height = output.size(2);
  int data_width = output.size(3);

  AVForwardLauncher(features, masks, kernel_size,
                            scale_factor, batch_size, num_channels, data_height,
                            data_width, output);

  return 1;
}

int av_backward_cuda(at::Tensor top_grad, at::Tensor features,
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

  int batch_size = top_grad.size(0);
  int num_channels = top_grad.size(1);
  int data_height = top_grad.size(2);
  int data_width = top_grad.size(3);

  AVBackwardLauncher(top_grad, features, masks, kernel_size,
                             scale_factor, batch_size, num_channels,
                             data_height, data_width, bottom_grad, mask_grad);

  return 1;
}