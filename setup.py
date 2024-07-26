# Copyright (c) 2020, Huawei Technologies.
# Copyright (c) 2019, NVIDIA CORPORATION.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import warnings
import os
import glob
import subprocess
from setuptools.command.build_ext import build_ext
from setuptools import setup, find_packages, Extension

import torch

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

cmdclass = {}
ext_modules = []

extras = {}

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

secure_compile_args = ['-fPIE', '-fPIC', '-fstack-protector-all', '-Wall', '-D__FILENAME__=\"$(notdir $(abspath $<))\"']

if (TORCH_MAJOR == 2 and TORCH_MINOR >= 1) or TORCH_MAJOR > 2 :
    secure_compile_args.append('-std=c++17')

secure_link_args = ['-Wl,-z,now', '-Wl,-z,relro', '-Wl,-z,noexecstack', '-s', '-Wl,--disable-new-dtags,--rpath']

def get_package_dir():
    if '--user' in sys.argv:
        package_dir = site.getusersitepackages()
    else:
        py_version = f'{sys.version_info.major}.{sys.version_info.minor}'
        package_dir = f'{sys.prefix}/lib/python{py_version}/site-packages'
    return package_dir


def CppExtension(name, sources, *args, **kwargs):
    r'''
    Creates a :class:`setuptools.Extension` for C++.
    '''
    package_dir = get_package_dir()
    temp_include_dirs = kwargs.get('include_dirs', [])
    temp_include_dirs.append(os.path.join(package_dir, 'torch/include'))
    temp_include_dirs.append(os.path.join(package_dir, 'torch/include/torch/csrc/api/include'))
    kwargs['include_dirs'] = temp_include_dirs

    temp_library_dirs = kwargs.get('library_dirs', [])
    temp_library_dirs.append(os.path.join(package_dir, 'torch/lib'))
    kwargs['library_dirs'] = temp_library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    libraries.append('torch_python')
    kwargs['libraries'] = libraries
    kwargs['language'] = 'c++'
    return Extension(name, sources, *args, **kwargs)


class BuildExtension(build_ext, object):

    def build_extensions(self):
        if self.compiler and '-Wstrict-prototypes' in self.compiler.compiler_so:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')

        if self.compiler and '-g' in self.compiler.compiler_so:
            self.compiler.compiler_so.remove('-g')

        return super(BuildExtension, self).build_extensions()


if "--pyprof" in sys.argv:
    string = "\n\nPyprof has been moved to its own dedicated repository and will " + \
             "soon be removed from Apex.  Please visit\n" + \
             "https://github.com/NVIDIA/PyProf\n" + \
             "for the latest version."
    warnings.warn(string, DeprecationWarning)
    with open('requirements.txt') as f:
        required_packages = f.read().splitlines()
        extras['pyprof'] = required_packages
    try:
        sys.argv.remove("--pyprof")
    except:
        pass
else:
    warnings.warn("Option --pyprof not specified. Not installing PyProf dependencies!")

if "--cpp_ext" in sys.argv or "--cuda_ext" in sys.argv:
    cmdclass['build_ext'] = BuildExtension

if "--cpp_ext" in sys.argv:
    sys.argv.remove("--cpp_ext")
    ext_modules.append(
        CppExtension('apex_C',
                     ['csrc/flatten_unflatten.cpp',],
                     extra_compile_args=secure_compile_args,
                     extra_link_args=secure_link_args))

    ext_modules.append(
        CppExtension('change_data_ptr',
                     ['csrc/combine_tensors/change_dataptr.cpp',],
                     extra_compile_args=secure_compile_args,
                     extra_link_args=secure_link_args))

if "--distributed_lamb" in sys.argv:
    cmdclass['build_ext'] = BuildExtension

if "--bnp" in sys.argv:
    cmdclass['build_ext'] = BuildExtension

if "--xentropy" in sys.argv:
    cmdclass['build_ext'] = BuildExtension

if "--deprecated_fused_adam" in sys.argv:
    cmdclass['build_ext'] = BuildExtension

if "--deprecated_fused_lamb" in sys.argv:
    cmdclass['build_ext'] = BuildExtension

if "--fast_multihead_attn" in sys.argv:
    cmdclass['build_ext'] = BuildExtension.with_options(use_ninja=False)

setup(
    name='apex',
    version='0.1-ascend',
    packages=find_packages(exclude=('build',
                                    'csrc',
                                    'include',
                                    'tests',
                                    'dist',
                                    'docs',
                                    'tests',
                                    'examples',
                                    'apex.egg-info',)),
    description='PyTorch Extensions written by NVIDIA',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    extras_require=extras,
)
