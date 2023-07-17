/*
   CUDA extension for SAPA
   by https://github.com/Teleppo
   modified from https://github.com/myownskyW7/CARAFE
*/
#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

using namespace at;  // temporal fix for pytorch<=0.4.1 (see #9848)

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65536;
  return min(optimal_block_num, max_block_num);
}

__device__ inline int Loc2Index(const int n, const int c, const int h,
                                const int w, const int channel_num,
                                const int height, const int width) {
  int index = w + (h + (c + n * channel_num) * height) * width;
  return index;
}
template <typename scalar_t>
__global__ void QKForward(const int nthreads,
                                   const scalar_t *bottom_data,
                                   const scalar_t *bottom_masks,
                                   const int kernel_size,
                                   const int scale_factor,
								   const int channels,
                                   const int height,
								   const int width,
                                   scalar_t *top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, k, ph, pw) is an element in the bottom_data
    int pw = index % width;
    int ph = (index / width) % height;
    int k = (index / width / height) % (kernel_size * kernel_size);
    int n = index / width / height / (kernel_size * kernel_size);

    int down_pw = pw / scale_factor;
    int down_ph = ph / scale_factor;
    int down_width = width / scale_factor;
    int down_height = height / scale_factor;
	int dix = k % kernel_size;
	int diy = k / kernel_size;
	int dh = down_ph - (kernel_size - 1) / 2 + diy;
	int dw = down_pw - (kernel_size - 1) / 2 + dix;

    scalar_t output_val = 0;
	for (int c = 0; c < channels; c++) {
        if (dh < 0 || dh > down_height - 1 || dw < 0 || dw > down_width - 1) {
            continue;
        }
		int feat_index =
            Loc2Index(n, c, dh, dw, channels, down_height, down_width);
        int mask_index =
            Loc2Index(n, c, ph, pw, channels, height, width);
		output_val += bottom_data[feat_index] * bottom_masks[mask_index];
	}
	top_data[index] = output_val;
  }
}


int QKForwardLauncher(const at::Tensor features,
						const at::Tensor masks,
                        const int kernel_size,
                        const int scale_factor,
						const int batch_size,
                        const int channels,
						const int height,
                        const int width,
						at::Tensor output) {
  const int output_size = batch_size * kernel_size * kernel_size * height * width;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.scalar_type(), "QKLauncherForward", ([&] {
        const scalar_t *bottom_data = features.data_ptr<scalar_t>();
        const scalar_t *bottom_masks = masks.data_ptr<scalar_t>();
        scalar_t *top_data = output.data_ptr<scalar_t>();

        QKForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_data, bottom_masks, kernel_size,
                scale_factor, channels, height, width, top_data);
      }));
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}

template <typename scalar_t>
__global__ void QKBackward(
    const int nthreads, const scalar_t *top_diff, const scalar_t *bottom_data,
    const scalar_t *bottom_masks, const int kernel_size,
    const int scale_factor, const int channels, const int height,
    const int width, scalar_t *bottom_diff, scalar_t *mask_diff) {
	CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, k, ph, pw) is an element in the bottom_data
    int pw = index % width;
    int ph = (index / width) % height;
    int k = (index / width / height) % (kernel_size * kernel_size);
    int n = index / width / height / (kernel_size * kernel_size);

    int down_pw = pw / scale_factor;
    int down_ph = ph / scale_factor;
    int down_width = width / scale_factor;
    int down_height = height / scale_factor;
	int dix = k % kernel_size;
	int diy = k / kernel_size;
	int dh = down_ph - (kernel_size - 1) / 2 + diy;
	int dw = down_pw - (kernel_size - 1) / 2 + dix;

	for (int c = 0; c < channels; c++) {
		if (dh < 0 || dh > down_height - 1 || dw < 0 || dw > down_width - 1) {
			  continue;
			}
		int feat_index =
            Loc2Index(n, c, dh, dw, channels, down_height, down_width);
        int mask_index =
            Loc2Index(n, c, ph, pw, channels, height, width);
        atomicAdd(bottom_diff + feat_index,
                  bottom_masks[mask_index] * top_diff[index]);
        atomicAdd(mask_diff + mask_index,
                  bottom_data[feat_index] * top_diff[index]);
	}
  }
}

int QKBackwardLauncher(const at::Tensor top_grad,
                        const at::Tensor features,
                        const at::Tensor masks,
						const int kernel_size,
						const int scale_factor,
						const int batch_size, const int channels,
						const int height, const int width,
						at::Tensor bottom_grad, at::Tensor mask_grad) {
  const int output_size = batch_size * kernel_size * kernel_size * height * width;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "QKLauncherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data_ptr<scalar_t>();
        const scalar_t *bottom_data = features.data_ptr<scalar_t>();
        const scalar_t *bottom_masks = masks.data_ptr<scalar_t>();
        scalar_t *bottom_diff = bottom_grad.data_ptr<scalar_t>();
        scalar_t *mask_diff = mask_grad.data_ptr<scalar_t>();

        QKBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, bottom_data, bottom_masks, kernel_size,
                scale_factor, channels, height, width, bottom_diff,
                mask_diff);
      }));

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}
