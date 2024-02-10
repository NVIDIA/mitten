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

from __future__ import annotations
from abc import ABC, abstractmethod, abstractclassmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, unique
from subprocess import CalledProcessError
from typing import Any, Callable, Dict, Final, Optional, List, Union

import json
import logging
import math
import os
import re
import shutil
import textwrap

from .base import Hardware
from .info_source import InfoSource, INFO_SOURCE_REGISTRY
from .accelerator import AcceleratorConfiguration
from .cpu import CPUConfiguration, CPU
from .memory import MemoryConfiguration
from ..utils import run_command
from ..registry import Registry


SYSTEM_JSON_MIG_MARKETING_NAME_FORMAT = re.compile(r"(.+) \((\d+)x(\d+)g\.(\d+)gb MIG\)")
"""re.Pattern: Regex to parse the marketing name for MIG systems, which are not consistent with nvidia-smi output or our
               internal system ID.

               Example matching strings:
                   "NVIDIA A100 (1x1g.5gb MIG)"
                   "A30 (12x1g.6gb MIG)"
                   "A100-SXM4-80GB (3x2g.20gb MIG)"

               match(1) - parent GPU name
               match(2) - number of MIG slices
               match(3) - GPCs per slice
               match(4) - mem per slice (in GB)
"""


@dataclass(eq=True, frozen=True)
class SystemConfiguration(Hardware):
    host_cpu_conf: CPUConfiguration
    """CPUConfiguration: CPU configuration of the host machine"""

    host_mem_conf: MemoryConfiguration
    """MemoryConfiguration: Memory configuration of the host machine"""

    accelerator_conf: AcceleratorConfiguration
    """AcceleratorConfiguration: Configuration of the accelerators in the system"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Dict[str, Any]: Optional metadata for this system configuration."""

    @classmethod
    def detect(cls) -> SystemConfiguration:
        """Consolidates the detected CPU, Memory, Accelerator, and NUMA configurations into a single object.

        Returns:
            SystemConfiguration: A SystemConfiguration object that contains all the detected components from runtime.
        """
        host_cpu_conf = CPUConfiguration.detect()
        host_mem_conf = MemoryConfiguration.detect()
        accelerator_conf = AcceleratorConfiguration.detect()
        return SystemConfiguration(host_cpu_conf, host_mem_conf, accelerator_conf)

    def __hash__(self) -> int:
        """Generate a hash from the components using prime numbers - See
        https://stackoverflow.com/questions/1145217/why-should-hash-functions-use-a-prime-number-modulus/1147232#1147232"""
        return hash(self.host_cpu_conf) + \
            13 * hash(self.host_mem_conf) + \
            29 * hash(self.accelerator_conf)

    def matches(self, other) -> bool:
        if other.__class__ == self.__class__:
            return self.host_cpu_conf == other.host_cpu_conf and \
                self.host_mem_conf == other.host_mem_conf and \
                self.accelerator_conf == other.accelerator_conf
        return NotImplemented

    def pretty_string(self) -> str:
        """Formatted, human-readable string displaying the data in the SystemConfiguration.

        Returns:
            str: 'Pretty-print' string representation of the SystemConfiguration
        """
        lines = ["SystemConfiguration:",
                 self.host_cpu_conf.pretty_string(),
                 self.host_mem_conf.pretty_string(),
                 self.accelerator_conf.pretty_string()]
        s = lines[0] + "\n" + textwrap.indent("\n".join(lines[1:]), " " * 4)
        return s


class SystemRegistry(Registry):
    """Used as a registry specifically for SystemConfigurations. This is useful for declaring a set of
    SystemConfigurations that are supported."""

    def __init__(self):
        super().__init__(SystemConfiguration)

    def get_match(self, sys_conf):
        """Matches a SystemConfiguration against all registered systems.
        """
        sys_definitions = self.registry.get_children()
        for node in sys_definitions:
            sdef = node.value
            if sdef == sys_conf:
                return node
        return None
