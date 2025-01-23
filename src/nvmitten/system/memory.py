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

import re

from .component import Component, Description
from ..constants import ByteSuffix
from ..memory import Memory


@dataclass(eq=True, frozen=True)
class HostMemory(Component):
    capacity: Memory

    @classmethod
    def detect(cls) -> Iterable[HostMemory]:
        with open("/proc/meminfo") as f:
            proc_meminfo = f.read().split("\n")
        mem_info = dict()
        for line in proc_meminfo:
            toks = re.split(r":\s*", line)
            if len(toks) == 2:
                mem_info[toks[0]] = toks[1]

        total = mem_info["MemTotal"].split()
        quantity = float(total[0])
        suff = ByteSuffix[total[1].upper()]
        capacity = Memory(quantity, suff)
        # Simplify to the largest possible unit for human readability
        mem = Memory.to_1000_base(capacity.to_bytes())
        return [HostMemory(mem)]

    def summary_description(self) -> Description:
        return Description(self.__class__,
                           capacity=self.capacity)

    def pretty_string(self) -> str:
        s = f"{self.capacity.pretty_string()} Host Memory"
        return s
