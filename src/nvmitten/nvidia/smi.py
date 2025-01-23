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

from typing import Dict, Final, List, Optional, Tuple
import os
import shutil
import subprocess

from .constants import ComputeSM
from .cupy import CUDAWrapper as cuda
from ..utils import run_command


class NvSMI:
    """Utility to interface with nvidia-smi
    """

    executable: Final[str] = "nvidia-smi"
    """str: The executable for desktop/server/non-SoC systems to interact with the GPU.
    For NVIDIA, this is nvidia-smi."""

    command: Final[str] = f"CUDA_VISIBLE_ORDER=PCI_BUS_ID {executable}"
    """str: The command to actually run. In this case, CUDA_VISIBLE_ORDER is hard set to PCI_BUS_ID for consistency."""

    @staticmethod
    def check_available() -> bool:
        """Checks if nvidia-smi is available on the system.

        Returns:
            bool: True if nvidia-smi is available, False if nvidia does not exist.
        """
        return shutil.which(NvSMI.executable) is not None

    @staticmethod
    def check_functional() -> bool:
        """Checks if nvidia-smi is functional.

        Returns:
            bool: True if nvidia-smi is functional. False if nvidia-smi is unavailable or fails to be executed. This is
                  the case if there is a faulty driver installation and nvidia-smi reports `NVIDIA-SMI has failed
                  because it couldn't communicate with the NVIDIA driver.`
        """
        if not NvSMI.check_available():
            return False
        try:
            run_command(NvSMI.command, get_output=False, tee=False, verbose=False)
            return True
        except subprocess.CalledProcessError:
            logging.warning(f"{NvSMI.command} exists but failed to execute")
            return False

    @staticmethod
    def query_gpu(fields: Tuple[str, ...], gpu_id: Optional[int] = None) -> List[Dict[str, str]]:
        """Executes an nvidia-smi --query-gpu command.

        Args:
            fields (Tuple[str, ...]): The fields to query for with nvidia-smi. See `nvidia-smi --help-query-gpu` for a
                                      full list of available fields.
            gpu_id (int): If set, queries only for the specified GPU. Otherwise, queries all available GPUs. (Default:
                          None)

        Returns:
            List[Dist[str, str]]: List of N dicts, where N is the number of GPU devices detected by nvidia-smi, each
                                  mapping the field to the value returned by nvidia-smi.
        """
        index_flag = ""
        if gpu_id is not None:
            index_flag = f"-i {gpu_id}"
        cmd = f"{NvSMI.command} {index_flag} --query-gpu={','.join(fields)} --format=csv,noheader"
        output = run_command(cmd, get_output=True, tee=False, verbose=False)
        return [dict(zip(fields, map(str.strip, line.split(','))))
                for line in output
                if len(line) > 0]  # Sometimes the last line is an empty string due to a newline

    @staticmethod
    def get_driver_version() -> str:
        """Returns the NVIDIA driver version installed.

        Returns:
            str: A string representing the NVIDIA driver version in the form <major>.<minor>.<patch>
        """
        # Driver version is consistent across all GPUs - arbitrarily pick GPU0
        info = NvSMI.query_gpu(("driver_version",), gpu_id=0)
        return info[0]["driver_version"]

    @staticmethod
    def check_mig_enabled(gpu_id: Optional[int] = None) -> bool:
        """Checks if MIG is enabled on a specific GPU. If no GPU ID is specified, returns whether or not *any* (i.e. at
        least 1) GPU has MIG enabled.

        Args:
            gpu_id (int): The ID of the GPU to check. If not specified, checks for any GPU. (Default: None)

        Returns:
            bool: Whether or not MIG is enabled for a given GPU (or any GPU if not specified)
        """
        info = NvSMI.query_gpu(("mig.mode.current",), gpu_id=gpu_id)
        return any(dat["mig.mode.current"] == "Enabled" for dat in info)

    @staticmethod
    def get_compute_sm(gpu_id: int) -> Tuple[int, int]:
        """Returns the compute capability of a given GPU. Note that older versions of nvidia-smi do not have support to
        query for this via --query-gpu, so there cuda-python is used as a fallback backend if this is the case.

        Args:
            gpu_id (int): The ID of the GPU to check.

        Returns:
            Tuple[int, int]: A tuple of (major, minor) version numbers
        """
        try:
            info = NvSMI.query_gpu(("compute_cap",), gpu_id=gpu_id)
            s = info[0]["compute_cap"]
            major, minor = map(int, s.split("."))
        except subprocess.CalledProcessError:
            # Force PyCUDA to be in PCI Bus order, which is the how gpu_ids are indexed in mitten.
            old_cuda_order = os.environ.get("CUDA_VISIBLE_ORDER", None)
            os.environ["CUDA_VISIBLE_ORDER"] = "PCI_BUS_ID"
            try:
                cuda.cuInit(0)
                device_attr_major = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
                major = cuda.cuDeviceGetAttribute(device_attr_major, gpu_id)
                device_attr_minor = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
                minor = cuda.cuDeviceGetAttribute(device_attr_minor, gpu_id)
            finally:
                if old_cuda_order is None:
                    del os.environ["CUDA_VISIBLE_ORDER"]
                else:
                    os.environ["CUDA_VISIBLE_ORDER"] = old_cuda_order
        return ComputeSM(major, minor)
