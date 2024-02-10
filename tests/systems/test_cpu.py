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

import pytest
from collections import namedtuple

from nvmitten.aliased_name import AliasedName
from nvmitten.constants import CPUArchitecture
from nvmitten.matchable import MATCH_ANY, MatchAllowList
from nvmitten.systems.cpu import *

from .utils import spoof_wrapper


ExpectedCPU = namedtuple("ExpectedCPU", ("name", "architecture", "core_count", "threads_per_core"))
nvidia_carmel = AliasedName("NVIDIA Carmel (ARMv8.2)", ("ARMv8 Processor rev 0 (v8l)",))


@pytest.mark.parametrize(
    "spoof_node_id,expected",
    [
        ("sample-system-1", ExpectedCPU("Neoverse-N1", CPUArchitecture.aarch64, 80, 1)),
        ("sample-system-2", ExpectedCPU(nvidia_carmel, CPUArchitecture.aarch64, 2, 1)),
        ("sample-system-3", ExpectedCPU("AMD EPYC 7742 64-Core Processor", CPUArchitecture.x86_64, 64, 1)),
    ]
)
def test_cpu_detect(spoof_node_id, expected):
    def _test():
        detected = CPU.detect()
        assert expected.name == detected.name
        assert expected.architecture == detected.architecture
        assert expected.core_count == detected.core_count
        assert expected.threads_per_core == detected.threads_per_core
    spoof_wrapper(spoof_node_id, ["CPU"], _test)


@pytest.mark.parametrize(
    "spoof_node_id,target,expected",
    [
        ("sample-system-1", ExpectedCPU("Neoverse-N1", CPUArchitecture.aarch64, MATCH_ANY, MATCH_ANY), False),
        ("sample-system-1", CPU("Neoverse-N1", CPUArchitecture.aarch64, MATCH_ANY, MATCH_ANY), True),
        ("sample-system-1", CPU("Neoverse-N1", CPUArchitecture.x86_64, MATCH_ANY, MATCH_ANY), False),
        ("sample-system-2", CPU(nvidia_carmel, CPUArchitecture.aarch64, MATCH_ANY, MATCH_ANY), True),
        ("sample-system-2", CPU(nvidia_carmel, CPUArchitecture.x86_64, MATCH_ANY, MATCH_ANY), False),
        ("sample-system-2", CPU(nvidia_carmel, CPUArchitecture.aarch64, MatchAllowList([1, 2]), 1), True),
        ("sample-system-2", CPU(nvidia_carmel, CPUArchitecture.aarch64, MatchAllowList([1, 3]), 1), False),
        ("sample-system-3", CPU("AMD EPYC 7742 64-Core Processor", CPUArchitecture.x86_64, MATCH_ANY, MATCH_ANY), True),
        ("sample-system-3", CPU("AMD EPYC 7742 64-Core Processor", CPUArchitecture.x86_64, MATCH_ANY, MATCH_ANY), True),
    ]
)
def test_cpu_match(spoof_node_id, target, expected):
    def _test():
        detected = CPU.detect()
        assert (target == detected) == expected
        assert (detected == target) == expected
    spoof_wrapper(spoof_node_id, ["CPU"], _test)


@pytest.mark.parametrize(
    "cpu",
    [
        (CPU("Neoverse-N1", CPUArchitecture.aarch64, MATCH_ANY, MATCH_ANY),),
        (CPU("Neoverse-N1", CPUArchitecture.aarch64, MATCH_ANY, MatchAllowList([1, 2, 4])),),
        (CPU("AMD EPYC 7742 64-Core Processor", CPUArchitecture.x86_64, 64, 1),),
    ]
)
def test_cpu_hashable(cpu):
    hash(cpu)  # Implicitly asserts that cpu is hashable by calling hash


@pytest.mark.parametrize(
    "spoof_node_id,expected",
    [
        ("sample-system-1", {CPU("Neoverse-N1", CPUArchitecture.aarch64, 80, 1): 1}),
        ("sample-system-2", {CPU("ARMv8 Processor rev 0 (v8l)", CPUArchitecture.aarch64, 2, 1): 4}),
        ("sample-system-3", {CPU("AMD EPYC 7742 64-Core Processor", CPUArchitecture.x86_64, 64, 1): 2}),
    ]
)
def test_cpu_configuration_detect(spoof_node_id, expected):
    def _test():
        detected = CPUConfiguration.detect()
        assert detected.layout == expected
    spoof_wrapper(spoof_node_id, ["CPU"], _test)


@pytest.mark.parametrize(
    "spoof_node_id,target,expected",
    [
        ("sample-system-1", CPUConfiguration({CPU("Neoverse-N1", CPUArchitecture.aarch64, 80, 1): 1}), True),
        ("sample-system-1", CPUConfiguration({CPU("Neoverse-N1", CPUArchitecture.aarch64, 80, 1): MATCH_ANY}), True),
        ("sample-system-1", CPUConfiguration({CPU("Neoverse-N1", CPUArchitecture.aarch64, 80, 1): 2}), False),
        ("sample-system-1", CPUConfiguration({CPU("Neoverse-N1", CPUArchitecture.aarch64, MATCH_ANY, MATCH_ANY): 1}), True),
        ("sample-system-2", CPUConfiguration({CPU(nvidia_carmel, CPUArchitecture.aarch64, 2, 1): 4}), True),
        ("sample-system-2", CPUConfiguration({CPU(nvidia_carmel, CPUArchitecture.aarch64, 2, 1): MatchAllowList([1, 2, 4])}), True),
        ("sample-system-2", CPUConfiguration({CPU(nvidia_carmel, CPUArchitecture.aarch64, 2, 1): MatchAllowList([1, 2, 3])}), False),
        ("sample-system-2", CPUConfiguration({CPU(nvidia_carmel, CPUArchitecture.aarch64, 2, 1): MATCH_ANY}), True),
        ("sample-system-2", CPUConfiguration({CPU(nvidia_carmel, CPUArchitecture.x86_64, 2, 1): 4}), False),
        ("sample-system-3", CPUConfiguration({CPU("AMD EPYC 7742 64-Core Processor", CPUArchitecture.x86_64, 64, 1): 2}), True),
        ("sample-system-3", CPUConfiguration({MATCH_ANY: 2}), True),
        ("sample-system-3", CPUConfiguration({MATCH_ANY: MATCH_ANY}), True),
        ("sample-system-3", CPUConfiguration({CPU("AMD EPYC 7742 64-Core Processor", CPUArchitecture.x86_64, 64, 1): MatchAllowList([1, 3, 4])}), False),
        ("sample-system-3", CPUConfiguration({CPU("AMD EPYC 7742 64-Core Processor", CPUArchitecture.x86_64, 64, 1): MatchAllowList([1, 2])}), True),
    ]
)
def test_cpu_configuration_match(spoof_node_id, target, expected):
    def _test():
        detected = CPUConfiguration.detect()
        assert (detected == target) == expected
        assert (target == detected) == expected
    spoof_wrapper(spoof_node_id, ["CPU"], _test)
