from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NVCC_ARGS = [
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
]

setup(
    name='sapa',
    version='0.0.1',
    license='L1',
    ext_modules=[
        CUDAExtension(
            'qk_ext', [
                'sapa/src/cuda/qk_cuda.cpp',
                'sapa/src/cuda/qk_cuda_kernel.cu',
                'sapa/src/qk_ext.cpp'
            ],
            define_macros=[('WITH_CUDA', None)],
            extra_compile_args={
                'cxx': [],
                'nvcc': NVCC_ARGS
            }),
        CUDAExtension(
            'av_ext', [
                'sapa/src/cuda/av_cuda.cpp',
                'sapa/src/cuda/av_cuda_kernel.cu',
                'sapa/src/av_ext.cpp'
            ],
            define_macros=[('WITH_CUDA', None)],
            extra_compile_args={
                'cxx': [],
                'nvcc': NVCC_ARGS
            })
    ],
    packages=find_packages(exclude=('test', )),
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
