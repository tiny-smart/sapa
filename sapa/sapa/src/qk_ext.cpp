/*
   CUDA extension for LCA
   by https://github.com/Poppuppy
*/
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
int qk_forward_cuda(at::Tensor features, at::Tensor masks,
                              int kernel_size, int scale_factor,
                              at::Tensor output);

int qk_backward_cuda(at::Tensor top_grad, at::Tensor features,
                               at::Tensor masks, int kernel_size,
                               int scale_factor,
                               at::Tensor bottom_grad, at::Tensor mask_grad);
#endif

int qk_forward(at::Tensor features, at::Tensor masks,
                         int kernel_size, int scale_factor,
                         at::Tensor output) {
  if (features.device().is_cuda()) {
#ifdef WITH_CUDA
    return qk_forward_cuda(features, masks, kernel_size,
        scale_factor, output);
#else
    AT_ERROR("qk is not compiled with GPU support");
#endif
  }
  AT_ERROR("qk is not implemented on CPU");
}

int qk_backward(at::Tensor top_grad, at::Tensor features,
                               at::Tensor masks, int kernel_size,
                               int scale_factor,
                               at::Tensor bottom_grad, at::Tensor mask_grad) {
  if (top_grad.device().is_cuda()) {
#ifdef WITH_CUDA
    return qk_backward_cuda(top_grad, features, masks, kernel_size,
    scale_factor, bottom_grad, mask_grad);
#else
    AT_ERROR("qk is not compiled with GPU support");
#endif
  }
  AT_ERROR("qk is not implemented on CPU");

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &qk_forward, "qk forward");
  m.def("backward", &qk_backward, "qk backward");
}
