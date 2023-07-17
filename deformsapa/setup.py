from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NVCC_ARGS = [
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
]

setup(
    name='deformsapa',
    version='0.0.1',
    license='MIT',
    ext_modules=[
        CUDAExtension(
            'dqk', [
                'deformsapa/src/cuda/deform_qk_cuda.cpp',
                'deformsapa/src/cuda/deform_qk_cuda_kernel.cu',
                'deformsapa/src/deform_qk.cpp'
            ],
            define_macros=[('WITH_CUDA', None)],
            extra_compile_args={
                'cxx': [],
                'nvcc': NVCC_ARGS
            }),
        CUDAExtension(
            'dav', [
                'deformsapa/src/cuda/deform_av_cuda.cpp',
                'deformsapa/src/cuda/deform_av_cuda_kernel.cu',
                'deformsapa/src/deform_av.cpp'
            ],
            define_macros=[('WITH_CUDA', None)],
            extra_compile_args={
                'cxx': [],
                'nvcc': NVCC_ARGS
            }),
    ],
    packages=find_packages(exclude=('test', )),
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
