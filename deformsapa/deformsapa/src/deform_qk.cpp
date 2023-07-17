/*
   CUDA extension for LCA
   by https://github.com/Poppuppy
*/
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
int deform_qk_forward_cuda(at::Tensor query,
						   at::Tensor key,
						   at::Tensor offset,
                           int num_point,
                           int scale_factor,
                           int groups,
						   at::Tensor output);

int deform_qk_backward_cuda(at::Tensor top_grad,
                            at::Tensor data_query,
                            at::Tensor data_key,
                            at::Tensor data_offset,
						    int num_point,
						    int scale_factor,
						    int groups,
						    at::Tensor grad_query,
						    at::Tensor grad_key,
						    at::Tensor grad_offset);
#endif

int deform_qk_forward(at::Tensor query,
						   at::Tensor key,
						   at::Tensor offset,
                           int num_point,
                           int scale_factor,
                           int groups,
						   at::Tensor output) {
  if (query.device().is_cuda()) {
#ifdef WITH_CUDA
    return deform_qk_forward_cuda(query, key, offset, num_point,
        scale_factor, groups, output);
#else
    AT_ERROR("deform_qk is not compiled with GPU support");
#endif
  }
  AT_ERROR("deform_qk is not implemented on CPU");
}

int deform_qk_backward(at::Tensor top_grad,
                            at::Tensor data_query,
                            at::Tensor data_key,
                            at::Tensor data_offset,
						    int num_point,
						    int scale_factor,
						    int groups,
						    at::Tensor grad_query,
						    at::Tensor grad_key,
						    at::Tensor grad_offset) {
  if (top_grad.device().is_cuda()) {
#ifdef WITH_CUDA
    return deform_qk_backward_cuda(top_grad, data_query, data_key, data_offset, num_point,
    scale_factor, groups, grad_query, grad_key, grad_offset);
#else
    AT_ERROR("deform_qk is not compiled with GPU support");
#endif
  }
  AT_ERROR("deform_qk is not implemented on CPU");

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &deform_qk_forward, "deform_qk forward");
  m.def("backward", &deform_qk_backward, "deform_qk backward");
}
