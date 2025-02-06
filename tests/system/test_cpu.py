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
from pathlib import Path
from unittest.mock import patch

from nvmitten.constants import CPUArchitecture
from nvmitten.interval import Interval
from nvmitten.system.cpu import *


@patch("nvmitten.system.cpu.run_command")
def test_cpu_detect(mock_run):
    with Path("tests/assets/system_detect_spoofs/cpu/sample-system-1").open() as f:
        contents = f.read().split("\n")

    mock_run.return_value = contents

    CPU.detect.cache_clear()
    cpu = CPU.detect()[0]
    mock_run.assert_called_once_with("lscpu -J", get_output=True, tee=False, verbose=False)
    assert cpu.name == "fake_cpu"
    assert cpu.architecture == CPUArchitecture.x86_64
    assert cpu.vendor == "DefinitelyRealVendor"
    assert cpu.cores_per_group == 64
    assert cpu.threads_per_core == 2
    assert cpu.n_groups == 2
    assert cpu.group_type == GroupType.Socket
    assert len(cpu.numa_nodes) == 8
    for i in range(8):
        assert cpu.numa_nodes[i] == [Interval(i * 16, (i + 1) * 16 - 1),
                                     Interval(i * 16 + 128, (i + 1) * 16 + 127)]


@patch("nvmitten.system.cpu.run_command")
def test_cpu_detect_missing_numa_node(mock_run):
    with Path("tests/assets/system_detect_spoofs/cpu/sample-system-2").open() as f:
        contents = f.read().split("\n")

    mock_run.return_value = contents

    CPU.detect.cache_clear()
    cpu = CPU.detect()[0]
    mock_run.assert_called_once_with("lscpu -J", get_output=True, tee=False, verbose=False)
    assert cpu.name == "Neoverse-V2"
    assert cpu.architecture == CPUArchitecture.aarch64
    assert cpu.vendor == "ARM"
    assert len(cpu.numa_nodes) == 18  # node2 and node10 are missing
    assert cpu.numa_nodes[2] is None
    assert cpu.numa_nodes[10] is None
    for idx in [3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]:
        assert cpu.numa_nodes[idx] == list()

    assert cpu.numa_nodes[0] == [Interval(0, 71)]
    assert cpu.numa_nodes[1] == [Interval(72, 143)]
