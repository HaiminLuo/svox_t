from setuptools import setup
import os.path as osp

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))

__version__ = None
exec(open('svox_t/version.py', 'r').read())

CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []

try:
    ext_modules = [
        CUDAExtension('svox_t.csrc', [
            'svox_t/csrc/svox.cpp',
            'svox_t/csrc/svox_kernel.cu',
            'svox_t/csrc/rt_kernel.cu',
            'svox_t/csrc/p2v_kernel.cu',
            'svox_t/csrc/quantizer.cpp',
        ], include_dirs=[osp.join(ROOT_DIR, "svox_t", "csrc", "include")],
        optional=True),
    ]
except:
    import warnings
    warnings.warn("Failed to build CUDA extension")
    ext_modules = []

setup(
    name='svox_t',
    version=__version__,
    author='Haimin Luo',
    author_email='luohm@shanghaitech.edu.cn',
    description='Skeleton driven dynamic sparse voxel N^3-tree data structure using CUDA',
    long_description='Skeleton driven dynamic sparse voxel N^3-tree data structure PyTorch extension, using CUDA',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    packages=['svox_t', 'svox_t.csrc'],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
