/*
   CUDA extension for DeformLCA
   by https://github.com/Poppuppy
*/
#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

using namespace at;  // temporal fix for pytorch<=0.4.1 (see #9848)

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
//   int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//   int max_block_num = 65536;
//   return min(optimal_block_num, max_block_num);
    return (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

template <typename scalar_t>
__device__ scalar_t deformable_im2col_bilinear(const scalar_t* bottom_data,
                                               const int height,
                                               const int width,
                                               const scalar_t h,
                                               const scalar_t w) {

  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
  {
    const int ptr1 = h_low * width + w_low;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (h_low >=0 && w_high <= width - 1)
  {
    const int ptr2 = h_low * width + w_high;
    v2 = bottom_data[ptr2];
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
  {
    const int ptr3 = h_high * width + w_low;
    v3 = bottom_data[ptr3];
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
  {
    const int ptr4 = h_high * width + w_high;
    v4 = bottom_data[ptr4];
  }

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__device__ void deformable_col2im_bilinear(const scalar_t* bottom_data,
                                           const int height,
                                           const int width,
                                           const scalar_t h,
                                           const scalar_t w,
                                           const scalar_t top_grad,
                                           const scalar_t query_weight,
                                           scalar_t* grad_query_ptr,
                                           scalar_t* grad_key_ptr,
                                           scalar_t* grad_offset_w_ptr,
                                           scalar_t* grad_offset_h_ptr)
{
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const scalar_t top_grad_value = top_grad * query_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
  {
    const int ptr1 = h_low * width + w_low;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_key_ptr+ptr1, w1*top_grad_value);
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
  {
    const int ptr2 = h_low * width + w_high;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_key_ptr+ptr2, w2*top_grad_value);
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
  {
    const int ptr3 = h_high * width + w_low;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_key_ptr+ptr3, w3*top_grad_value);
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
  {
    const int ptr4 = h_high * width + w_high;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_key_ptr+ptr4, w4*top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  atomicAdd(grad_query_ptr, top_grad * val);
  atomicAdd(grad_offset_w_ptr, grad_w_weight * top_grad_value);
  atomicAdd(grad_offset_h_ptr, grad_h_weight * top_grad_value);
}

template <typename scalar_t>
__global__ void DeformAVForward(const int n,
                                const scalar_t* data_attn,
                                const scalar_t* data_value,
                                const scalar_t* data_offset,
                                const int num_point,
                                const int scale_factor,
                                const int groups,
                                const int channels,
                                const int height,
                                const int width,
                                scalar_t* data_out) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int qw = _temp % width;
    _temp /= width;
    const int qh = _temp % height;
    _temp /= height;
    const int c = _temp % channels;
    _temp /= channels;
    const int b = _temp;

    const int attn_channels = groups * num_point;
    const int attn_group = c / (channels / groups);

    const int down_qw = qw / scale_factor;
    const int down_qh = qh / scale_factor;
    const int key_width = width / scale_factor;
    const int key_height = height / scale_factor;

    int data_attn_ptr = ((b * attn_channels + attn_group * num_point) * height + qh) * width + qw;
    const int data_attn_stride = height * width;
    int data_offset_ptr = ((b * attn_channels + attn_group * num_point << 1) * height + qh) * width + qw;
    const int data_half_offset_stride = height * width;
    const int data_offset_stride = data_half_offset_stride << 1;

    const scalar_t* data_value_ptr = data_value + (b * channels + c) * key_height * key_width;

	scalar_t output_val = 0;
	for (int ip = 0; ip < num_point; ++ip) {
        const scalar_t offset_w = data_offset[data_offset_ptr];
        const scalar_t offset_h = data_offset[data_offset_ptr + data_half_offset_stride];
        const scalar_t weight = data_attn[data_attn_ptr];
        const scalar_t kh = down_qh + offset_h;
        const scalar_t kw = down_qw + offset_w;

        if (kh > -1 && kh < key_height && kw > -1 && kw < key_width) {
                output_val += deformable_im2col_bilinear(data_value_ptr, key_height, key_width, kh, kw) * weight;
            }

        data_attn_ptr += data_attn_stride;
        data_offset_ptr += data_offset_stride;

	}
	data_out[index] = output_val;
  }
}

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
						    at::Tensor output) {
  const int output_size = batch_size * channels * height * width;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      attn.scalar_type(), "DeformQKLauncherForward", ([&] {
        const scalar_t *bottom_attn = attn.data_ptr<scalar_t>();
        scalar_t *bottom_value = value.data_ptr<scalar_t>();
        const scalar_t *bottom_offset = offset.data_ptr<scalar_t>();
        scalar_t *top_data = output.data_ptr<scalar_t>();

        DeformAVForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_attn, bottom_value, bottom_offset, num_point,
                scale_factor, groups, channels, height, width, top_data);
      }));
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}

template <typename scalar_t>
__global__ void DeformAVBackward(const int n,
                                const scalar_t* top_grad,
                                const scalar_t* data_attn,
                                const scalar_t* data_value,
                                const scalar_t* data_offset,
                                const int num_point,
                                const int scale_factor,
                                const int groups,
                                const int channels,
                                const int height,
                                const int width,
                                scalar_t *grad_attn,
                                scalar_t *grad_value,
                                scalar_t *grad_offset) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int qw = _temp % width;
    _temp /= width;
    const int qh = _temp % height;
    _temp /= height;
    const int c = _temp % channels;
    _temp /= channels;
    const int b = _temp;

    const int attn_channels = num_point * groups;
    const int attn_group = c / (channels / groups);

    const int down_qw = qw / scale_factor;
    const int down_qh = qh / scale_factor;
    const int key_width = width / scale_factor;
    const int key_height = height / scale_factor;

    int data_attn_ptr = ((b * attn_channels + attn_group * num_point) * height + qh) * width + qw;
    const int data_attn_stride = height * width;
    int data_offset_ptr = ((b * attn_channels + attn_group * num_point << 1) * height + qh) * width + qw;
    const int data_half_offset_stride = height * width;
    const int data_offset_stride = data_half_offset_stride << 1;
    const scalar_t* data_value_ptr = data_value + (b * channels + c) * key_height * key_width;

    grad_attn += data_attn_ptr;
    grad_offset += data_offset_ptr;
    grad_value += (b * channels + c) * key_height * key_width;

    for (int ip = 0; ip < num_point; ++ip) {
        const scalar_t offset_w = data_offset[data_offset_ptr];
        const scalar_t offset_h = data_offset[data_offset_ptr + data_half_offset_stride];
        const scalar_t kh = down_qh + offset_h;
        const scalar_t kw = down_qw + offset_w;

        if (kh > -1 && kh < key_height && kw > -1 && kw < key_width) {
                deformable_col2im_bilinear(data_value_ptr, key_height, key_width, kh, kw,
                top_grad[index], data_attn[data_attn_ptr],
                grad_attn, grad_value, grad_offset, grad_offset + data_half_offset_stride);
            }

        data_attn_ptr += data_attn_stride;
        data_offset_ptr += data_offset_stride;
        grad_attn += data_attn_stride;
        grad_offset += data_offset_stride;
	}
  }
}

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
						     at::Tensor grad_offset) {
  const int output_size = batch_size * channels * height * width;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "DeformAVLauncherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data_ptr<scalar_t>();
        const scalar_t *bottom_attn = data_attn.data_ptr<scalar_t>();
        scalar_t *bottom_value = data_value.data_ptr<scalar_t>();
        const scalar_t *bottom_offset = data_offset.data_ptr<scalar_t>();
        scalar_t *diff_attn = grad_attn.data_ptr<scalar_t>();
        scalar_t *diff_value = grad_value.data_ptr<scalar_t>();
        scalar_t *diff_offset = grad_offset.data_ptr<scalar_t>();

        DeformAVBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, bottom_attn, bottom_value, bottom_offset, num_point,
                scale_factor, groups, channels, height, width, diff_attn, diff_value, diff_offset);
      }));

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}
