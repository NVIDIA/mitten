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


import numpy as np
import pytest

from nvmitten.nvidia.smi import NvSMI
if not NvSMI.check_functional():
    pytest.skip("Skipping GPU-only tests", allow_module_level=True)

from nvmitten.nvidia.cupy import (
    CUDAWrapper as cuda,
    CUDARTWrapper as cudart,
    HostDeviceBuffer,
)


def test_error_handling():
    ptr = cudart.cudaMallocHost(1)
    cudart.cudaFreeHost(ptr)

    # Double free will always throw an invalid argument error
    with pytest.raises(cudart.Error.cudaErrorInvalidValue):
        cudart.cudaFreeHost(ptr)


class TestHostDeviceBuffer:

    def test_create(self):
        buff = HostDeviceBuffer((2, 4, 5), np.float16, name="abc")

        assert buff.shape == (2, 4, 5)
        assert buff.nptype == np.float16
        assert buff.name == "abc"
        assert buff.host != None and buff.host != 0
        assert buff.device != None and buff.device != 0
        assert buff._owns_host
        assert buff._owns_device
        assert buff._bytesize == (2 * 4 * 5 * 2)

    def test_create_device_only(self):
        buff = HostDeviceBuffer((2, 4, 5), np.float16, name="abc", host=0)

        assert buff.shape == (2, 4, 5)
        assert buff.nptype == np.float16
        assert buff.name == "abc"
        assert buff.host == 0
        assert buff.device != None and buff.device != 0
        assert not buff._owns_host
        assert buff._owns_device
        assert buff._bytesize == (2 * 4 * 5 * 2)

    def test_create_host_only(self):
        buff = HostDeviceBuffer((2, 4, 5), np.float16, name="abc", device=0)

        assert buff.shape == (2, 4, 5)
        assert buff.nptype == np.float16
        assert buff.name == "abc"
        assert buff.host != None and buff.host != 0
        assert buff.device == 0
        assert buff._owns_host
        assert not buff._owns_device
        assert buff._bytesize == (2 * 4 * 5 * 2)

    def test_ndarray_interface_copied(self, cuda_stream):
        buff = HostDeviceBuffer((2, 4, 3), np.float32, name="ndarray_test")

        x = np.arange(24, dtype=np.float32)
        np.random.shuffle(x)
        x = x.reshape((2, 4, 3))

        buff.ndarray_to_host(x, cuda_stream, sync=True)
        y = buff.host_to_ndarray(copy=True)

        assert np.allclose(x, y)

        # Doing a new copy should not overwrite y
        z = np.arange(24, dtype=np.float32).reshape((2, 4, 3))
        while np.allclose(x, z):
            z = z.flatten()
            np.random.shuffle(z)
            z = z.reshape((2, 4, 3))

        buff.ndarray_to_host(z, cuda_stream, sync=True)
        assert not np.allclose(y, z)

    def test_ndarray_interface_nocopy(self, cuda_stream):
        buff = HostDeviceBuffer((2, 4, 3), np.float32, name="ndarray_test")

        x = np.arange(24, dtype=np.float32)
        np.random.shuffle(x)
        x = x.reshape((2, 4, 3))

        buff.ndarray_to_host(x, cuda_stream, sync=True)
        y = buff.host_to_ndarray(copy=False)

        assert np.allclose(x, y)

        # Doing a new copy should overwrite y
        z = np.arange(24, dtype=np.float32).reshape((2, 4, 3))
        while np.allclose(x, z):
            z = z.flatten()
            np.random.shuffle(z)
            z = z.reshape((2, 4, 3))

        buff.ndarray_to_host(z, cuda_stream, sync=True)
        assert np.allclose(y, z)
