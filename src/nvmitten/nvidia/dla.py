# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
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


import ctypes
import pathlib
import glob

# Find .so file for cudla
_so_files = glob.glob(str((pathlib.Path(__file__).parent / "cudla.*.so").resolve()))
if len(_so_files) == 1:  # There's only supposed to be 1 .so. For safety, disable DLA if more than 1 .so is found
    cudla = ctypes.CDLL(_so_files[0])
else:
    cudla = None
