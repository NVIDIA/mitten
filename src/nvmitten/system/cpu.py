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
from enum import Enum, unique
from functools import cached_property
from typing import Dict, Set, Iterable

import json

from .component import Component, Description
from ..constants import CPUArchitecture
from ..interval import Interval
from ..tree import Tree
from ..utils import run_command


@unique
class GroupType(Enum):
    Socket = "socket"
    Cluster = "cluster"


@dataclass(eq=True, frozen=True)
class CPU(Component):
    name: str
    architecture: CPUArchitecture
    vendor: str
    cores_per_group: int
    threads_per_core: int
    n_groups: int
    group_type: GroupType = GroupType.Socket
    numa_nodes: List[List[Interval]] = field(default_factory=list)
    flags: Set[str] = field(default_factory=set)
    vulnerabilities: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def detect(cls) -> Iterable[CPU]:
        lscpu_out = run_command("lscpu -J", get_output=True, tee=False, verbose=False)
        cpu_info = json.loads(" ".join(lscpu_out))

        # lscpu has different JSON formats depending on version. Newer versions use nesting and have a "children" key
        # for more structured data.
        assert len(cpu_info) == 1 and "lscpu" in cpu_info and type(cpu_info["lscpu"]) is list, \
            "Unexpected lscpu output format."
        cpu_info = cpu_info["lscpu"]

        # Make lscpu output more indexable
        _t = Tree("lscpu", None)

        def process_field(d, parents=None):
            field = d["field"].rstrip(' ').rstrip(':').lower()
            data = d["data"]

            if parents is None:
                keyspace = [field]
            else:
                keyspace = parents + [field]

            _t[keyspace] = data
            if "children" in d:
                for child in d["children"]:
                    process_field(child, parents=keyspace)

        for d in cpu_info:
            process_field(d)

        # In older versions, "Model name" will be a top level key. In newer versions it under "Vendor ID"
        is_old_lscpu = (len(_t["vendor id"].children) == 0)
        if is_old_lscpu:
            try:
                name = _t["model name"].value
            except KeyError:
                name = "Unnamed CPU"

            architecture = CPUArchitecture(_t["architecture"].value)
            vendor = _t["vendor id"].value

            # Sockets vs. Clusters. lscpu will group cpu core info in either sockets or clusters.
            try:
                group_type = GroupType.Socket
                n_groups = int(_t["socket(s)"].value)
                cores_per_group = int(_t["core(s) per socket"].value)
            except (ValueError, KeyError):
                group_type = GroupType.Cluster
                n_groups = int(_t["cluster(s)"].value)
                cores_per_group = int(_t["core(s) per cluster"].value)

            threads_per_core = int(_t["thread(s) per core"].value)

            numa_nodes = []
            if "numa node(s)" in _t.children:
                n_numa_nodes = int(_t["numa node(s)"].value)
                _valid = 0
                i = 0
                while _valid < n_numa_nodes:
                    if i >= 256:
                        raise RuntimeError("Impossible state reached while parsing NUMA info")

                    k = f"numa node{i} cpu(s)"
                    i += 1
                    if k in _t.children:
                        _valid += 1
                        if _t[k].value:
                            numa_nodes.append(Interval.interval_list_from_str(_t[k].value))
                        else:
                            numa_nodes.append(list())
                    else:
                        numa_nodes.append(None)
            flags = set(_t["flags"].value.split(" "))

            vulns = dict()
            for child in _t.get_children():
                if child.name.startswith("vulnerability "):
                    if child.value == "Not affected":
                        continue
                    vulns[child.name[len("vulnerability "):]] = child.value
        else:
            try:
                name = _t["vendor id", "model name"].value
            except KeyError:
                name = "Unnamed CPU"

            architecture = CPUArchitecture(_t["architecture"].value)
            vendor = _t["vendor id"].value
            if "socket(s)" in _t["vendor id"].children or "cluster(s)" in _t["vendor id"].children:
                try:
                    group_type = GroupType.Socket
                    n_groups = int(_t["vendor id", "socket(s)"].value)
                    cores_per_group = int(_t["vendor id", "core(s) per socket"].value)
                except (ValueError, KeyError):
                    group_type = GroupType.Cluster
                    n_groups = int(_t["vendor id", "cluster(s)"].value)
                    cores_per_group = int(_t["vendor id", "core(s) per cluster"].value)
                threads_per_core = int(_t["vendor id", "thread(s) per core"].value)
                flags = set(_t["vendor id", "flags"].value.split(" "))
            else:
                try:
                    group_type = GroupType.Socket
                    n_groups = int(_t["vendor id", "model name", "socket(s)"].value)
                    cores_per_group = int(_t["vendor id", "model name", "core(s) per socket"].value)
                except (ValueError, KeyError):
                    group_type = GroupType.Cluster
                    n_groups = int(_t["vendor id", "model name", "cluster(s)"].value)
                    cores_per_group = int(_t["vendor id", "model name", "core(s) per cluster"].value)
                threads_per_core = int(_t["vendor id", "model name", "thread(s) per core"].value)
                flags = set(_t["vendor id", "model name", "flags"].value.split(" "))


            numa_nodes = []
            if "numa" in _t.children:
                n_numa_nodes = int(_t["numa", "numa node(s)"].value)
                i = 0
                _valid = 0
                while _valid < n_numa_nodes:
                    if i >= 256:
                        raise RuntimeError("Impossible state reached while parsing NUMA info")
                    k = f"numa node{i} cpu(s)"
                    i += 1
                    if k in _t["numa"].children:
                        _valid += 1
                        if _t["numa", k].value:
                            numa_nodes.append(Interval.interval_list_from_str(_t["numa", k].value))
                        else:
                            numa_nodes.append(list())
                    else:
                        numa_nodes.append(None)

            vulns = dict()
            if "vulnerabilities" in _t.children:
                for vuln in _t["vulnerabilities"].get_children():
                    if vuln.value == "Not affected":
                        continue
                    vulns[vuln.name] = vuln.value

        return [CPU(name,
                    architecture,
                    vendor,
                    cores_per_group,
                    threads_per_core,
                    n_groups,
                    group_type=group_type,
                    numa_nodes=numa_nodes,
                    flags=flags,
                    vulnerabilities=vulns)]

    @cached_property
    def active_cores(self) -> Set[int]:
        taskset = run_command("taskset -c -p $$", get_output=True, tee=False, verbose=False)
        active_cpus_str = taskset[0].split(": ")[-1]
        active_cpus = set()
        for interval in Interval.interval_list_from_str(active_cpus_str):
            active_cpus = active_cpus.union(interval.to_set())
        return active_cpus

    @cached_property
    def n_threads(self) -> int:
        return self.n_groups * self.cores_per_group * self.threads_per_core

    def summary_description(self) -> Description:
        return Description(self.__class__,
                           architecture=self.architecture,
                           vendor=self.vendor)

    def pretty_string(self) -> str:
        s = f"{self.name} ({self.n_threads} Threads, {self.architecture.valstr()})"
        return s
