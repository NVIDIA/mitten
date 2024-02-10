# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
import pytest
from collections import namedtuple

from nvmitten.constants import ByteSuffix
from nvmitten.matchable import MATCH_ANY
from nvmitten.memory import Memory
from nvmitten.systems.info_source import INFO_SOURCE_REGISTRY
from nvmitten.systems.memory import MemoryConfiguration, get_mem_info


MEM_INFO_SOURCE = INFO_SOURCE_REGISTRY.get("memory")


@pytest.mark.parametrize(
    "spoof_node_id,expected",
    [
        ("sample-system-4", Memory(32616612, ByteSuffix.KB)),
        ("sample-system-3", Memory(990640728, ByteSuffix.KB))
    ]
)
def test_mem_detect(spoof_node_id, expected):
    try:
        mem_info_filepath = os.path.join(os.getcwd(), "tests", "assets", "system_detect_spoofs", "mem", spoof_node_id)

        def _override_mem_info():
            return get_mem_info(sys_meminfo_file=mem_info_filepath)
        MEM_INFO_SOURCE.fn = _override_mem_info
        MEM_INFO_SOURCE.reset(hard=True)

        detected = MemoryConfiguration.detect()
        assert detected.host_memory_capacity == expected
    finally:
        MEM_INFO_SOURCE.fn = get_mem_info
        MEM_INFO_SOURCE.reset(hard=True)


@pytest.mark.parametrize(
    "spoof_node_id,target,expected",
    [
        ("sample-system-4", MemoryConfiguration(Memory(32, ByteSuffix.GB)), True),
        ("sample-system-4", MemoryConfiguration(Memory(31, ByteSuffix.GiB)), True),
        ("sample-system-4", MemoryConfiguration(Memory(30, ByteSuffix.GB)), False),
        ("sample-system-3", MemoryConfiguration(Memory(1, ByteSuffix.TB)), True),
        ("sample-system-3", MATCH_ANY, True),
        ("sample-system-3", MemoryConfiguration(Memory(900, ByteSuffix.GB)), False),
        ("sample-system-3", MemoryConfiguration(Memory(999, ByteSuffix.GB)), True)  # 999GB is close enough to 1TB to allow.
    ]
)
def test_mem_match(spoof_node_id, target, expected):
    try:
        mem_info_filepath = os.path.join(os.getcwd(), "tests", "assets", "system_detect_spoofs", "mem", spoof_node_id)

        def _override_mem_info():
            return get_mem_info(sys_meminfo_file=mem_info_filepath)
        MEM_INFO_SOURCE.fn = _override_mem_info
        MEM_INFO_SOURCE.reset(hard=True)

        detected = MemoryConfiguration.detect()
        assert (detected == target) == expected
        assert (target == detected) == expected
    finally:
        MEM_INFO_SOURCE.fn = get_mem_info
        MEM_INFO_SOURCE.reset(hard=True)
