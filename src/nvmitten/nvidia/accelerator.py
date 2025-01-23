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
from functools import cached_property
from typing import List

import textwrap

from .constants import ComputeSM
from .cupy import CUDAWrapper as cuda
from .dla import cudla
from .smi import NvSMI
from ..interval import NumericRange
from ..memory import Memory
from ..system.accelerator import Accelerator, NUMASupported
from ..system.component import Description


@dataclass
class GPU(NUMASupported, Accelerator):
    pci_id: str
    compute_sm: ComputeSM
    vram: Memory
    max_power_limit: float
    is_integrated: bool
    gpu_index: int = None

    @cached_property
    def numa_host_id(self) -> int:
        if self.gpu_index is None:
            raise IndexError("gpu_index is not set")

        nhid = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
                                         self.gpu_index)
        return None if nhid < 0 else nhid

    @cached_property
    def uuid(self) -> str:
        s = cuda.cuDeviceGetUuid_v2(self.gpu_index).bytes.hex()
        # Follow same format as NvSMI
        return f"GPU-{s[:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:32]}"

    @property
    def power_limit(self) -> int:
        d = NvSMI.query_gpu(("power.limit",), gpu_id=self.gpu_index)
        return float(d[0]["power.limit"].split()[0])

    @classmethod
    def detect(cls) -> List[GPU]:
        cuda.cuInit(0)
        gpus = []

        # I tried to use only cuda-python for this since it seemed to have some nice features, but I found they were
        # giving inconsistent results. For instance, PCI_DEVICE_ID was always returning as 0, and some options like
        # CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED are reported differently by cuda and cudart.
        # Using nvidia-smi for the majority of fields, but we can explore using cuda-python exclusively if it gets
        # updated.
        data = None
        if NvSMI.check_functional():
            data = NvSMI.query_gpu(("pci.device_id",
                                    "power.max_limit"))

        n_gpus = cuda.cuDeviceGetCount()
        for i in range(n_gpus):
            # Get GPU name. Surely 128 bytes is enough right?
            gpu_name = cuda.cuDeviceGetName(128, i).decode("utf-8").strip("\x00 ")
            mem_bytes = cuda.cuDeviceTotalMem(i)
            vram = Memory.to_1024_base(mem_bytes)

            sm_major = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, i)
            sm_minor = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, i)
            compute_sm = ComputeSM(sm_major, sm_minor)

            is_integrated = bool(cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_INTEGRATED, i))

            # Figure out how to derive these without NvSMI
            pci_device_id = None if data is None else data[i]["pci.device_id"]
            if data is None or (pml := data[i]["power.max_limit"]) == "[N/A]":
                max_power_limit = None
            else:
                max_power_limit = float(pml.split()[0])

            gpus.append(GPU(gpu_name,
                            pci_device_id,
                            compute_sm,
                            vram,
                            max_power_limit,
                            is_integrated,
                            gpu_index=i))
        return gpus

    def summary_description(self) -> Description:
        _mpl_desc = None
        if self.max_power_limit:
            _mpl_desc = NumericRange(self.max_power_limit, rel_tol=0.05)

        return Description(self.__class__,
                           pci_id=self.pci_id,
                           vram=NumericRange(self.vram, rel_tol=0.05),
                           max_power_limit=_mpl_desc,
                           is_integrated=self.is_integrated)

    def pretty_string(self) -> str:
        lines = [f"{self.name} (PCI_ID: {self.pci_id})",
                 f"Memory Capacity: {self.vram.pretty_string()}",
                 f"Max Power Limit: {self.max_power_limit} W",
                 f"Compute Capability: {int(self.compute_sm)}",
                 f"Is Integrated Graphics: {self.is_integrated}"]
        return lines[0] + '\n' + textwrap.indent('\n'.join(lines[1:]), ' ' * 4)


@dataclass
class DLA(Accelerator):
    core_id: int

    @classmethod
    def detect(cls) -> List[DLA]:
        if cudla is None:
            return list()

        return [DLA("nvdla", i)
                for i in range(cudla.getNbDLACores())]

    def summary_description(self) -> Description:
        return Description(self.__class__)

    def pretty_string(self) -> str:
        return f"NvDLA (Core {self.core_id})"
