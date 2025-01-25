#!/usr/bin/env python
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools.command.build_ext import build_ext
from setuptools import Extension, setup
import os


class NVCCBuildExt(build_ext):
    """Overrides the compiler settings to use NVCC, since we are compiling CUDA binaries. Default C Flags that
    setuptools always injects are not compatible with NVCC.
    """

    BINARY_PATH = "/usr/local/cuda/bin/nvcc"

    def build_extensions(self):
        if NVCCBuildExt.hasdla():
            self.compiler.set_executable("compiler_so", NVCCBuildExt.BINARY_PATH)
            self.compiler.set_executable("compiler_cxx", NVCCBuildExt.BINARY_PATH)
            self.compiler.set_executable("linker_so", NVCCBuildExt.BINARY_PATH)
        build_ext.build_extensions(self)

    @staticmethod
    def hasdla():
        return os.path.exists(NVCCBuildExt.BINARY_PATH) and os.path.exists("/usr/local/cuda/include/cudla.h")


if __name__ == "__main__":
    # Build DLA extension only if nvcc and cudla.h are present
    define_macros = []
    libraries = []
    extra_compile_args = []
    extra_link_args = ["--shared"]
    if NVCCBuildExt.hasdla():
        define_macros.append(("HASDLA", 1))
        libraries.append("cudla")
        extra_compile_args.extend(["-Xcompiler", "-fPIC"])
    dla_extension = Extension("nvmitten.nvidia.cudla",
                              sources=["src/nvmitten/nvidia/dla.cpp"],
                              define_macros=define_macros,
                              libraries=libraries,
                              extra_compile_args=extra_compile_args,
                              extra_link_args=extra_link_args,
                              optional=True)

    setup(ext_modules=[dla_extension],
          cmdclass={"build_ext": NVCCBuildExt})
