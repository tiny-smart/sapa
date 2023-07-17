#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
int av_forward_cuda(at::Tensor features, at::Tensor masks,
                              int kernel_size, int scale_factor,
                              at::Tensor output);

int av_backward_cuda(at::Tensor top_grad, at::Tensor features,
                               at::Tensor masks, int kernel_size,
                               int scale_factor,
                               at::Tensor bottom_grad, at::Tensor mask_grad);
#endif

int av_forward(at::Tensor features, at::Tensor masks,
                         int kernel_size, int scale_factor,
                         at::Tensor output) {
  if (features.device().is_cuda()) {
#ifdef WITH_CUDA
    return av_forward_cuda(features, masks, kernel_size,
        scale_factor, output);
#else
    AT_ERROR("av is not compiled with GPU support");
#endif
  }
  AT_ERROR("av is not implemented on CPU");
}

int av_backward(at::Tensor top_grad, at::Tensor features,
                               at::Tensor masks, int kernel_size,
                               int scale_factor,
                               at::Tensor bottom_grad, at::Tensor mask_grad) {
  if (top_grad.device().is_cuda()) {
#ifdef WITH_CUDA
    return av_backward_cuda(top_grad, features, masks, kernel_size,
        scale_factor, bottom_grad, mask_grad);
#else
    AT_ERROR("av is not compiled with GPU support");
#endif
  }
  AT_ERROR("av is not implemented on CPU");

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &av_forward, "av forward");
  m.def("backward", &av_backward, "av backward");
}