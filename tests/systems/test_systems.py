# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Unit tests for code/common/systems/systems.py. Tests for the expected functionality of the base classes and methods."""

import os
import pytest
from collections import namedtuple

from nvmitten.systems.systems import *
from nvmitten.systems.cpu import CPUConfiguration
from nvmitten.systems.memory import MemoryConfiguration, get_mem_info
from nvmitten.systems.accelerator import AcceleratorConfiguration
from nvmitten.nvidia.accelerator import *  # Load NVIDIA accelerator module

from .utils import spoof_wrapper


@pytest.mark.parametrize(
    ("spoof_node_id", "sm_value"),
    [
        ("sample-system-7", 80),
        ("sample-system-5", 80),
    ]
)
def test_system_configuration(spoof_node_id, sm_value):
    # We assume that CPU/Memory/AcceleratorConfiguration are functional from their own individual tests.
    # Compare the SystemConfiguration with each of the components
    def _setup():
        def _override_accelerator_info():
            return get_accelerator_info()
        INFO_SOURCE_REGISTRY.get("nvidia_smi").fn = _override_accelerator_info

        def _override_gpu_info():
            return get_gpu_info(skip_sm_check=True, force_sm_value=sm_value)
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvgpu")).fn = _override_gpu_info

        def _override_mig_info():
            return get_mig_info(skip_sm_check=True, force_sm_value=sm_value)
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvmig")).fn = _override_mig_info

        mem_info_filepath = os.path.join(os.getcwd(), "tests", "assets", "system_detect_spoofs", "mem", spoof_node_id)

        def _override_mem_info():
            return get_mem_info(sys_meminfo_file=mem_info_filepath)
        INFO_SOURCE_REGISTRY.get("memory").fn = _override_mem_info

    def _cleanup():
        INFO_SOURCE_REGISTRY.get("nvidia_smi").fn = get_accelerator_info
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvgpu")).fn = get_gpu_info
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvmig")).fn = get_mig_info
        INFO_SOURCE_REGISTRY.get("memory").fn = get_mem_info

    def _test():
        system = SystemConfiguration.detect()
        cpu_conf = CPUConfiguration.detect()
        gpu_conf = AcceleratorConfiguration.detect()
        mem_conf = MemoryConfiguration.detect()
        assert system.host_cpu_conf == cpu_conf
        assert system.host_mem_conf == mem_conf
        assert system.accelerator_conf == gpu_conf
        assert system == SystemConfiguration(cpu_conf, mem_conf, gpu_conf)

    spoof_wrapper(spoof_node_id,
                  [("CPU",), ("nvidia_smi",), ("accelerators", "nvgpu"), ("accelerators", "nvmig"), ("memory",)],
                  _test,
                  setup=_setup,
                  cleanup=_cleanup)
