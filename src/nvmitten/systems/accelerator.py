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
from dataclasses import dataclass
from enum import Enum, unique
from subprocess import CalledProcessError
from typing import Any, Callable, ClassVar, Final, List, Tuple, Union

import logging
import os
import re
import shutil
import textwrap

from .base import Hardware
from ..aliased_name import AliasedName
from ..constants import ByteSuffix, AcceleratorType
from ..matchable import Matchable
from .info_source import InfoSource, INFO_SOURCE_REGISTRY
from ..utils import run_command



SOC_MODEL_FILEPATH: Final[str] = "/sys/firmware/devicetree/base/model"
"""str: Defines the OS file that contains the SoC's name"""


@dataclass(eq=True, frozen=True)
class Accelerator(Hardware):
    name: Union[Matchable, str, AliasedName]
    accelerator_type: Union[Matchable, AcceleratorType]


@dataclass(eq=True, frozen=True)
class AcceleratorConfiguration(Hardware):
    layout: Dict[Accelerator, int]

    def __hash__(self):
        return hash(frozenset(self.layout))

    def get_accelerators(self):
        return list(self.layout.keys())

    def get_primary_accelerator(self):
        if len(self.layout) == 0:
            return None
        return self.get_accelerators()[0]

    @classmethod
    def detect(cls) -> AcceleratorConfiguration:
        """Detects possible known accelerator types and builds a map of accelerator -> count. Known accelerator types
        must be implemented subclasses of 'Hardware'.

        Returns:
            AcceleratorConfiguration: An AcceleratorConfiguration from runtime data
        """
        layout = defaultdict(int)

        # Detect GPUs first
        try:
            xlr_infosrcs = INFO_SOURCE_REGISTRY.get("accelerators")
        except KeyError:
            # No accelerators have been registered, so none can be detected.
            return None
        for infosrc_node in xlr_infosrcs:
            infosrc = infosrc_node.value
            infosrc.reset()
            while infosrc.has_next():
                xlr = infosrc.klass.detect()
                layout[xlr] += 1

        return AcceleratorConfiguration(layout)

    def matches(self, other) -> bool:
        if other.__class__ == self.__class__:
            if len(self.layout) != len(other.layout):
                return False

            for accelerator, count in self.layout.items():
                # We actually have to iterate through each accelerator in other in case of Matchables. hashes are not guaranteed
                # to be equivalent even if a.matches(b).
                found = False
                for other_accelerator, other_count in other.layout.items():
                    if accelerator == other_accelerator:
                        found = True
                        if count != other_count:
                            return False
                        break
                if not found:
                    return False
            return True
        return NotImplemented

    def pretty_string(self) -> str:
        """Formatted, human-readable string displaying the data in the AcceleratorConfiguration.

        Returns:
            str: 'Pretty-print' string representation of the AcceleratorConfiguration
        """
        lines = ["AcceleratorConfiguration:"]
        for accelerator, count in self.layout.items():
            lines.append(f"{count}x " + accelerator.pretty_string())
        s = lines[0] + "\n" + textwrap.indent("\n".join(lines[1:]), " " * 4)
        return s
