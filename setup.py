from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='seeanythingfar',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=[
        # Template for when we add custom CUDA ops (e.g., fast voxelization)
        # CUDAExtension(
        #     name='seeanythingfar.ops._ext',
        #     sources=[
        #         'src/seeanythingfar/ops/src/voxelize.cpp',
        #         'src/seeanythingfar/ops/src/voxelize_cuda.cu',
        #     ],
        # )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)