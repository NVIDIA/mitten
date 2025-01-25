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
from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar, List, Type

from .component import Component


@dataclass
class Accelerator(Component):
    _registered: ClassVar[List] = []

    name: str

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        Accelerator._registered.append(cls)

    @classmethod
    def detect(cls) -> Dict[Type[Accelerator], List[Accelerator]]:
        detected = dict()
        for c in Accelerator._registered:
            detected[c] = c.detect()
        return detected


class NUMASupported:
    """Mixin class to help with attaching accelerators to host NUMA nodes.
    """

    @property
    def numa_host_id(self) -> int:
        """The host NUMA node ID affiliated with this device.

        Returns:
            int: None if NUMA is supported but not enabled. Otherwise returns the host NUMA node ID as an int.
        """
        return None
