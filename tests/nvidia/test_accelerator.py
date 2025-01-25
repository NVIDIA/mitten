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
import pytest
from collections import namedtuple
from unittest.mock import MagicMock, patch

from nvmitten.nvidia.smi import NvSMI
if not NvSMI.check_functional():
    pytest.skip("Skipping GPU-only tests", allow_module_level=True)

from nvmitten.memory import Memory
from nvmitten.nvidia.constants import ComputeSM
from nvmitten.nvidia.cupy import CUDAWrapper as cuda
from nvmitten.nvidia.accelerator import *


@patch("nvmitten.nvidia.accelerator.NvSMI.query_gpu")
@patch("nvmitten.nvidia.accelerator.NvSMI.check_functional")
@patch("nvmitten.nvidia.accelerator.cuda.cuDeviceGetUuid_v2")
@patch("nvmitten.nvidia.accelerator.cuda.cuDeviceTotalMem")
@patch("nvmitten.nvidia.accelerator.cuda.cuDeviceGetName")
@patch("nvmitten.nvidia.accelerator.cuda.cuInit")
@patch("nvmitten.nvidia.accelerator.cuda.cuDeviceGetCount")
@patch("nvmitten.nvidia.accelerator.cuda.cuDeviceGetAttribute")
def test_gpu_detect(mock_getattr,
                    mock_getcount,
                    mock_init,
                    mock_getname,
                    mock_totalmem,
                    mock_uuid,
                    mock_nvsmi_func,
                    mock_nvsmi_query):
    def getattrspoof(attr, idx):
        assert idx == 0

        if attr == cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
            return 4
        elif attr == cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
            return 2
        elif attr == cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_INTEGRATED:
            return True
        elif attr == cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID:
            return 1337

    mock_getattr.side_effect = getattrspoof
    mock_getcount.return_value = 1
    mock_init.return_value = None
    mock_getname.return_value = "Dummy GPU (TM)".encode("utf-8")
    mock_totalmem.return_value = 10 << 20
    FakeUUID = namedtuple("FakeUUID", ["bytes"])
    mock_uuid.return_value = FakeUUID(bytes=b"\x12\x34\x56\x78\x12\x34\x12\x34\x12\x34\x12\x34\x56\x78\x90\xAB")
    mock_nvsmi_func.return_value = True
    mock_nvsmi_query.return_value = [{"pci.device_id": "0xDEADBEEF", "power.max_limit": "99999 W"}]

    gpus = GPU.detect()
    assert len(gpus) == 1
    assert gpus[0].name == "Dummy GPU (TM)"
    assert gpus[0].pci_id == "0xDEADBEEF"
    assert gpus[0].compute_sm == ComputeSM(4, 2)
    assert gpus[0].vram._num_bytes == 10 << 20
    assert gpus[0].max_power_limit == 99999
    assert gpus[0].is_integrated
    assert gpus[0].gpu_index == 0
    assert gpus[0].numa_host_id == 1337
    assert gpus[0].uuid == "GPU-12345678-1234-1234-1234-1234567890ab"
