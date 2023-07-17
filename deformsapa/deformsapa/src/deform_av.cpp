/*
   CUDA extension for LCA
   by https://github.com/Poppuppy
*/
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
int deform_av_forward_cuda(at::Tensor attn,
						   at::Tensor value,
						   at::Tensor offset,
                           int num_point,
                           int scale_factor,
                           int groups,
						   at::Tensor output);

int deform_av_backward_cuda(at::Tensor top_grad,
                            at::Tensor data_attn,
                            at::Tensor data_value,
                            at::Tensor data_offset,
						    int num_point,
						    int scale_factor,
						    int groups,
						    at::Tensor grad_attn,
						    at::Tensor grad_value,
						    at::Tensor grad_offset);
#endif

int deform_av_forward(at::Tensor attn,
						   at::Tensor value,
						   at::Tensor offset,
                           int num_point,
                           int scale_factor,
                           int groups,
						   at::Tensor output) {
  if (attn.device().is_cuda()) {
#ifdef WITH_CUDA
    return deform_av_forward_cuda(attn, value, offset, num_point,
        scale_factor, groups, output);
#else
    AT_ERROR("deform_av is not compiled with GPU support");
#endif
  }
  AT_ERROR("deform_av is not implemented on CPU");
}

int deform_av_backward(at::Tensor top_grad,
                            at::Tensor data_attn,
                            at::Tensor data_value,
                            at::Tensor data_offset,
						    int num_point,
						    int scale_factor,
						    int groups,
						    at::Tensor grad_attn,
						    at::Tensor grad_value,
						    at::Tensor grad_offset) {
  if (top_grad.device().is_cuda()) {
#ifdef WITH_CUDA
    return deform_av_backward_cuda(top_grad, data_attn, data_value, data_offset, num_point,
    scale_factor, groups, grad_attn, grad_value, grad_offset);
#else
    AT_ERROR("deform_av is not compiled with GPU support");
#endif
  }
  AT_ERROR("deform_av is not implemented on CPU");

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &deform_av_forward, "deform_av forward");
  m.def("backward", &deform_av_backward, "deform_av backward");
}
