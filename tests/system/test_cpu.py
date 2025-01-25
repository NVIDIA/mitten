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
