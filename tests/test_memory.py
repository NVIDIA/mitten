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


import pytest
import random

from nvmitten.constants import ByteSuffix
from nvmitten.memory import Memory
from nvmitten.json_utils import loads, dumps


@pytest.mark.parametrize(
    "mem,expected_bytes",
    [
        (Memory(2.7, ByteSuffix.GB), 2700000000.0),
        (Memory(4.5, ByteSuffix.MiB), 4718592.0),
        (Memory(1000, ByteSuffix.B), 1000),
    ]
)
def test_memory_to_bytes(mem, expected_bytes):
    assert mem.to_bytes() == expected_bytes


@pytest.mark.parametrize(
    "b,method,expected_mem",
    [
        (2700000000.0, Memory.to_1000_base, Memory(2.7, ByteSuffix.GB)),
        (1000000.0, Memory.to_1000_base, Memory(1.0, ByteSuffix.MB)),
        (1024 ** 4, Memory.to_1024_base, Memory(1.0, ByteSuffix.TiB)),
        (3.14 * 1024 ** 2, Memory.to_1024_base, Memory(3.14, ByteSuffix.MiB)),
    ]
)
def test_bytes_to_memory(b, method, expected_mem):
    assert method(b) == expected_mem


@pytest.mark.parametrize(
    "s,expected_mem",
    [
        ("1.5TB", Memory(1.5, ByteSuffix.TB)),
        ("7 GB", Memory(7, ByteSuffix.GB)),
        ("12KiB", Memory(12, ByteSuffix.KiB)),
        ("15 gb", None),
        ("-7 GB", None),
        ("0.0 MB", Memory(0, ByteSuffix.MB)),
        ("1.5TB asdf", None),
        ("22.7", None),
    ]
)
def test_memory_from_string(s, expected_mem):
    if expected_mem is None:
        with pytest.raises(ValueError):
            Memory.from_string(s)
    else:
        assert Memory.from_string(s) == expected_mem

@pytest.mark.parametrize(
    "mem,o,expected,raised",
    [
        (Memory(1.0, ByteSuffix.KB), 5, Memory.to_1000_base(6 * 1000), None),
        (Memory(1.0, ByteSuffix.KB), Memory(2, ByteSuffix.MB), Memory.to_1000_base(2001000), None),
        (Memory(1.0, ByteSuffix.KB), "asdf", None, TypeError),
        (Memory(1.0, ByteSuffix.KB), Memory(1, ByteSuffix.KiB), Memory.to_1000_base(2024), None),
    ]
)
def test_memory_add(mem, o, expected, raised):
    if raised is not None:
        with pytest.raises(raised):
            mem + o
    else:
        assert (mem + o) == expected

@pytest.mark.parametrize(
    "mem,o,expected,raised",
    [
        (Memory(2.0, ByteSuffix.KB), .5, Memory.to_1000_base(1000), None),
        (Memory(1.0, ByteSuffix.KB), Memory(2, ByteSuffix.MB), None, TypeError),
        (Memory(1.0, ByteSuffix.KB), "asdf", None, TypeError),
        (Memory(2.0, ByteSuffix.KB), 16, Memory.to_1000_base(32000), None),
    ]
)
def test_memory_mul(mem, o, expected, raised):
    if raised is not None:
        with pytest.raises(raised):
            mem * o
    else:
        assert (mem * o) == expected

def test_memory_json():
    # Generate some random memory object
    num_bytes = random.randint(1, 1e12)
    m_1000 = Memory.to_1000_base(num_bytes)

    s = dumps(m_1000)
    new_m = loads(s)
    assert m_1000 == new_m
    assert new_m.byte_suffix.value[0] == 1000
