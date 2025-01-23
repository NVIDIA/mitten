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

from nvmitten.system.memory import *


def _load_asset(system_id):
    with Path(f"tests/assets/system_detect_spoofs/mem/sample-system-{system_id}").open() as f:
        contents = f.read()
    return contents


@pytest.fixture
def sample_systems():
    # Preload contents before pyfakefs takes over the filesystem.
    return {1: (_load_asset(1), 990640728000),
            2: (_load_asset(2), 32616612000),
            3: (_load_asset(3), 990640724000),
            4: (_load_asset(4), 528216216000)}


def test_hostmemory_detect_1(sample_systems, fs):
    contents, nbytes = sample_systems[1]

    fs.create_file("/proc/meminfo", contents=contents)
    HostMemory.detect.cache_clear()
    mem = HostMemory.detect()[0]
    assert mem.capacity._num_bytes == nbytes


def test_hostmemory_detect_2(sample_systems, fs):
    contents, nbytes = sample_systems[2]

    fs.create_file("/proc/meminfo", contents=contents)
    HostMemory.detect.cache_clear()
    mem = HostMemory.detect()[0]
    assert mem.capacity._num_bytes == nbytes


def test_hostmemory_detect_3(sample_systems, fs):
    contents, nbytes = sample_systems[3]

    fs.create_file("/proc/meminfo", contents=contents)
    HostMemory.detect.cache_clear()
    mem = HostMemory.detect()[0]
    assert mem.capacity._num_bytes == nbytes


def test_hostmemory_detect_4(sample_systems, fs):
    contents, nbytes = sample_systems[4]

    fs.create_file("/proc/meminfo", contents=contents)
    HostMemory.detect.cache_clear()
    mem = HostMemory.detect()[0]
    assert mem.capacity._num_bytes == nbytes
