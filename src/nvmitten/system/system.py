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
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Type

import textwrap

from .component import Component, Description
from .cpu import CPU
from .accelerator import Accelerator, NUMASupported
from .memory import HostMemory
from ..interval import Interval
from ..json_utils import ClassKeyedDict


@dataclass
class NUMANode:
    index: int
    cpu_cores: List[Interval]
    accelerators: Dict[Type[Accelerator], List[Accelerator]] = field(default_factory=dict)


@dataclass
class System(Component):
    cpu: CPU
    host_memory: HostMemory
    accelerators: Dict[Type[Component], List[Component]]
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def detect(cls) -> Iterable[System]:
        cpu = CPU.detect()[0]
        host_memory = HostMemory.detect()[0]
        accelerators = Accelerator.detect()

        return [System(cpu,
                       host_memory,
                       accelerators)]

    @cached_property
    def n_accelerators(self) -> int:
        return sum(len(v) for k, v in self.accelerators.items())

    def get_numa_config(self) -> List[NUMANode]:
        active_cpus = self.cpu.active_cores

        config = list()
        for i, intervals in enumerate(self.cpu.numa_nodes):
            if intervals is None:
                continue

            cpu_set = set()
            for interval in intervals:
                cpu_set = cpu_set.union(interval.to_set())

            invalid_cpus = cpu_set - active_cpus
            valid_cpus = cpu_set.intersection(active_cpus)

            if len(invalid_cpus) > 0:
                logging.warning(f"CPUs {invalid_cpus} are in reported NUMA affinities but not in taskset. Removing.")
                intervals = Interval.build_interval_list(valid_cpus)

            if len(intervals) == 0:
                raise ValueError(f"No CPUs in taskset for NUMA node {i}")

            config.append(NUMANode(i, intervals))

        for accelerator_t, devices in self.accelerators.items():
            if not issubclass(accelerator_t, NUMASupported):
                continue

            for device in devices:
                if device.numa_host_id is None:
                    continue
                _d = config[device.numa_host_id].accelerators
                if accelerator_t not in _d:
                    _d[accelerator_t] = list()
                _d[accelerator_t].append(device)

        return config

    def summary_description(self) -> Description:
        accelerator_summary = {_t: [a.summary_description() for a in L]
                               for _t, L in self.accelerators.items()}

        return Description(System,
                           cpu=self.cpu.summary_description(),
                           host_memory=self.host_memory.summary_description(),
                           accelerators=ClassKeyedDict(accelerator_summary),
                           extras=deepcopy(self.extras))

    def pretty_string(self) -> str:
        lines = ["System",
                 self.cpu.pretty_string(),
                 self.host_memory.pretty_string()]
        for _t, L in self.accelerators.items():
            if len(L) == 0:
                continue
            lines.append(f"{_t.__module__}.{_t.__name__}")
            for a in L:
                lines.append(textwrap.indent(a.pretty_string(), ' ' * 4))
        return lines[0] + '\n' + textwrap.indent('\n'.join(lines[1:]), ' ' * 4)
