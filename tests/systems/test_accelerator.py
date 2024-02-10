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
import os
import shutil
import pytest
from collections import namedtuple

from nvmitten.systems.accelerator import *
from nvmitten.systems.info_source import INFO_SOURCE_REGISTRY
from nvmitten.nvidia.accelerator import *
from nvmitten.constants import AcceleratorType, ByteSuffix
from nvmitten.memory import Memory

from .utils import spoof_wrapper


ExpectedGPU = namedtuple("ExpectedGPU", (
    "name",
    "accelerator_type",
    "vram",
    "max_power_limit",
    "pci_id",
    "compute_sm",
))


@pytest.mark.parametrize(
    "spoof_node_id,expected",
    [
        ("sample-system-1", [
            ExpectedGPU(
                "NVIDIA A100 80GB PCIe",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(81251, ByteSuffix.MiB).to_bytes()),
                300.0,
                "0x20B510DE",
                None
            )
            for _ in range(2)
        ]),
        ("sample-system-3", [
            ExpectedGPU(
                "NVIDIA A30",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(24258, ByteSuffix.MiB).to_bytes()),
                165.0,
                "0x20B710DE",
                None
            )
            for _ in range(8)
        ]),
        ("sample-system-5", []),
        ("sample-system-6", [
            ExpectedGPU(
                "NVIDIA A100-SXM4-40GB",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(40536, ByteSuffix.MiB).to_bytes()),
                400.0,
                "0x20B010DE",
                None
            )
            for _ in range(1)
        ]),
        ("sample-system-7", [
            ExpectedGPU(
                "NVIDIA A100-SXM-80GB",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(81251, ByteSuffix.MiB).to_bytes()),
                275.0,
                "0x20B210DE",
                None
            )
            for _ in range(4)
        ]),
    ]
)
def test_desktop_gpu_detect(spoof_node_id, expected):
    def _test():
        gpus = []
        while INFO_SOURCE_REGISTRY.get(("accelerators", "nvgpu")).has_next():
            gpus.append(GPU.detect())

        assert len(gpus) == len(expected)
        for gpu_i, expected_i in zip(gpus, expected):
            assert expected_i.name == gpu_i.name
            assert expected_i.accelerator_type == gpu_i.accelerator_type
            assert expected_i.vram == gpu_i.vram
            assert expected_i.max_power_limit == gpu_i.max_power_limit
            assert expected_i.pci_id == gpu_i.pci_id
            assert expected_i.compute_sm == gpu_i.compute_sm

    def _setup():
        def _override_gpu_info():
            return get_gpu_info(skip_sm_check=True)
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvgpu")).fn = _override_gpu_info

    def _cleanup():
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvgpu")).fn = get_gpu_info

    spoof_wrapper(spoof_node_id, [("accelerators", "nvgpu")], _test, setup=_setup, cleanup=_cleanup)


@pytest.mark.parametrize(
    "spoof_node_id,expected",
    [
        ("sample-system-2", [
            ExpectedGPU(
                "Jetson-AGX",
                AcceleratorType.Integrated,
                None,
                None,
                None,
                None
            )
        ]),
        ("sample-system-8", [
            ExpectedGPU(
                "NVIDIA Jetson Xavier NX Developer Kit",
                AcceleratorType.Integrated,
                None,
                None,
                None,
                None
            )
        ]),
    ]
)
def test_soc_gpu_detect(spoof_node_id, expected):
    # To detect as an SoC, nvidia-smi has to be unavailable, and we must override SOC_MODEL_FILEPATH
    try:
        soc_model_filepath = os.path.join(os.getcwd(), "tests", "assets", "system_detect_spoofs", "gpu", spoof_node_id)

        def _override_gpu_info():
            return get_gpu_info(soc_model_filepath=soc_model_filepath, force_no_gpu_cmd=True, skip_sm_check=True)
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvgpu")).fn = _override_gpu_info
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvgpu")).reset(hard=True)

        gpus = []
        while INFO_SOURCE_REGISTRY.get(("accelerators", "nvgpu")).has_next():
            gpus.append(GPU.detect())

        assert len(gpus) == len(expected)
        for gpu_i, expected_i in zip(gpus, expected):
            assert expected_i.name == gpu_i.name
            assert expected_i.accelerator_type == gpu_i.accelerator_type
            assert expected_i.vram == gpu_i.vram
            assert expected_i.max_power_limit == gpu_i.max_power_limit
            assert expected_i.pci_id == gpu_i.pci_id
            assert expected_i.compute_sm == gpu_i.compute_sm
    finally:
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvgpu")).fn = get_gpu_info


@pytest.mark.parametrize(
    "spoof_node_id,visible_devices,expected",
    [
        ("sample-system-3", "", [
            ExpectedGPU(
                "NVIDIA A30",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(24258, ByteSuffix.MiB).to_bytes()),
                165.0,
                "0x20B710DE",
                None
            )
            for _ in range(8)
        ]),
        ("sample-system-3", "GPU-f68752fb-b8cc-b6f3-5cf7-bbece3a34636", [
            ExpectedGPU(
                "NVIDIA A30",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(24258, ByteSuffix.MiB).to_bytes()),
                165.0,
                "0x20B710DE",
                None
            )
        ]),
        ("sample-system-3", "0,2,4,6", [
            ExpectedGPU(
                "NVIDIA A30",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(24258, ByteSuffix.MiB).to_bytes()),
                165.0,
                "0x20B710DE",
                None
            )
            for _ in range(4)
        ]),
        ("sample-system-3", "1,GPU-337430c5-a027-d44f-11b7-13753ea341ad", [
            ExpectedGPU(
                "NVIDIA A30",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(24258, ByteSuffix.MiB).to_bytes()),
                165.0,
                "0x20B710DE",
                None
            )
            for _ in range(2)
        ]),
        ("sample-system-7", "0,GPU-8f652e12-ba71-5354-68cd-0330da3b3a24", [
            ExpectedGPU(
                "NVIDIA A100-SXM-80GB",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(81251, ByteSuffix.MiB).to_bytes()),
                275.0,
                "0x20B210DE",
                None
            )
            for _ in range(1)  # 0 and UUID are the same GPU
        ]),
        ("sample-system-7", "GPU-8f652e12-ba71-5354-68cd-0330da3b3a24,GPU-e22be74c-a7bd-7667-1020-57f4922fa2a9,2", [
            ExpectedGPU(
                "NVIDIA A100-SXM-80GB",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(81251, ByteSuffix.MiB).to_bytes()),
                275.0,
                "0x20B210DE",
                None
            )
            for _ in range(2)
        ]),
    ]
)
def test_cuda_visible_devices(spoof_node_id, visible_devices, expected):
    def _test():
        gpus = []
        while INFO_SOURCE_REGISTRY.get(("accelerators", "nvgpu")).has_next():
            gpus.append(GPU.detect())

        assert len(gpus) == len(expected)
        for gpu_i, expected_i in zip(gpus, expected):
            assert expected_i.name == gpu_i.name
            assert expected_i.accelerator_type == gpu_i.accelerator_type
            assert expected_i.vram == gpu_i.vram
            assert expected_i.max_power_limit == gpu_i.max_power_limit
            assert expected_i.pci_id == gpu_i.pci_id
            assert expected_i.compute_sm == gpu_i.compute_sm

    def _setup():
        def _override_gpu_info():
            return get_gpu_info(skip_sm_check=True)
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvgpu")).fn = _override_gpu_info

    def _cleanup():
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvgpu")).fn = get_gpu_info

    spoof_wrapper(spoof_node_id, [("accelerators", "nvgpu")], _test, visible_devices=visible_devices, setup=_setup,
                  cleanup=_cleanup)


@pytest.mark.parametrize(
    "spoof_node_id,target,expected",
    [
        ("sample-system-1", [
            GPU(
                "NVIDIA A100 80GB PCIe",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(81251, ByteSuffix.MiB).to_bytes()),
                300.0,
                "0x20B510DE",
                None
            )
            for _ in range(2)
        ], True),
        ("sample-system-1", [
            GPU(
                "NVIDIA A100 80GB PCIe",
                AcceleratorType.Integrated,
                Memory.to_1024_base(Memory(81251, ByteSuffix.MiB).to_bytes()),
                300.0,
                "0x20B510DE",
                None
            )
            for _ in range(2)
        ], False),
        ("sample-system-3", [
            GPU(
                "NVIDIA A30",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(24258, ByteSuffix.MiB).to_bytes()),
                165.0,
                "0x20B710DE",
                None
            )
            for _ in range(8)
        ], True),
        ("sample-system-3", [
            GPU(
                "NVIDIA A10",
                AcceleratorType.Integrated,
                Memory.to_1024_base(Memory(24258, ByteSuffix.MiB).to_bytes()),
                165.0,
                "0x20B710DE",
                None
            )
            for _ in range(8)
        ], False),
        ("sample-system-6", [
            GPU(
                "NVIDIA A100-SXM4-40GB",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(40536, ByteSuffix.MiB).to_bytes()),
                400.0,
                "0x20B010DE",
                None
            )
            for _ in range(1)
        ], True),
        ("sample-system-6", [
            GPU(
                "NVIDIA A100-SXM4-40GB",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(40530, ByteSuffix.MiB).to_bytes()),
                400.0,
                "0x20B010DE",
                None
            )
            for _ in range(1)
        ], False),
        ("sample-system-7", [
            GPU(
                "NVIDIA A100-SXM-80GB",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(81251, ByteSuffix.MiB).to_bytes()),
                275.0,
                "0x20B210DE",
                None
            )
            for _ in range(4)
        ], True),
        ("sample-system-7", [
            GPU(
                "NVIDIA A100-SXM-80GB",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(81251, ByteSuffix.MiB).to_bytes()),
                400.0,
                "0x20B210DE",
                None
            )
            for _ in range(4)
        ], False),
    ]
)
def test_gpu_match(spoof_node_id, target, expected):
    def _test():
        gpus = []
        while INFO_SOURCE_REGISTRY.get(("accelerators", "nvgpu")).has_next():
            gpus.append(GPU.detect())
        assert (target == gpus) == expected
        assert (gpus == target) == expected

    def _setup():
        def _override_gpu_info():
            return get_gpu_info(skip_sm_check=True)
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvgpu")).fn = _override_gpu_info

    def _cleanup():
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvgpu")).fn = get_gpu_info

    spoof_wrapper(spoof_node_id, [("accelerators", "nvgpu")], _test, setup=_setup, cleanup=_cleanup)


ExpectedMIG = namedtuple("ExpectedGPU", (
    "name",
    "accelerator_type",
    "vram",
    "max_power_limit",
    "pci_id",
    "compute_sm",
    "num_gpcs",
))


@pytest.mark.parametrize(
    "spoof_node_id,expected",
    [
        ("sample-system-5", [
            ExpectedMIG(
                "NVIDIA A30 MIG-1g.6gb",
                AcceleratorType.Discrete,
                Memory(6, ByteSuffix.GiB),
                165.0,
                "0x20B710DE",
                None,
                1,
            )
            for _ in range(32)
        ]),
        ("sample-system-3", []),
    ]
)
def test_mig_detect(spoof_node_id, expected):
    def _test():
        migs = []
        while INFO_SOURCE_REGISTRY.get(("accelerators", "nvmig")).has_next():
            migs.append(MIG.detect())

        assert len(migs) == len(expected)
        for mig_i, expected_i in zip(migs, expected):
            assert expected_i.name == mig_i.name
            assert expected_i.accelerator_type == mig_i.accelerator_type
            assert expected_i.vram == mig_i.vram
            assert expected_i.max_power_limit == mig_i.max_power_limit
            assert expected_i.pci_id == mig_i.pci_id
            assert expected_i.compute_sm == mig_i.compute_sm
            assert expected_i.num_gpcs == mig_i.num_gpcs

    def _setup():
        def _override_accelerator_info():
            return get_accelerator_info()
        INFO_SOURCE_REGISTRY.get("nvidia_smi").fn = _override_accelerator_info

        def _override_mig_info():
            return get_mig_info(skip_sm_check=True)
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvmig")).fn = _override_mig_info

    def _cleanup():
        INFO_SOURCE_REGISTRY.get("nvidia_smi").fn = get_accelerator_info
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvmig")).fn = get_mig_info

    spoof_wrapper(spoof_node_id, [("nvidia_smi",), ("accelerators", "nvmig")], _test, setup=_setup, cleanup=_cleanup)


@pytest.mark.parametrize(
    "spoof_node_id,target,expected",
    [
        ("sample-system-5", [
            MIG(
                "NVIDIA A30 MIG-1g.6gb",
                AcceleratorType.Discrete,
                Memory(6, ByteSuffix.GiB),
                165.0,
                "0x20B710DE",
                None,
                1,
            )
            for _ in range(32)
        ], True),
        ("sample-system-3", [], True),
    ]
)
def test_mig_match(spoof_node_id, target, expected):
    def _test():
        migs = []
        while INFO_SOURCE_REGISTRY.get(("accelerators", "nvmig")).has_next():
            migs.append(MIG.detect())
        assert (target == migs) == expected
        assert (migs == target) == expected

    def _setup():
        def _override_accelerator_info():
            return get_accelerator_info()
        INFO_SOURCE_REGISTRY.get("nvidia_smi").fn = _override_accelerator_info

        def _override_mig_info():
            return get_mig_info(skip_sm_check=True)
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvmig")).fn = _override_mig_info

    def _cleanup():
        INFO_SOURCE_REGISTRY.get("nvidia_smi").fn = get_accelerator_info
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvmig")).fn = get_mig_info

    spoof_wrapper(spoof_node_id, [("nvidia_smi",), ("accelerators", "nvmig")], _test, setup=_setup, cleanup=_cleanup)


@pytest.mark.parametrize(
    "spoof_node_id,expected",
    [
        ("sample-system-1", {
            GPU("NVIDIA A100 80GB PCIe",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(81251, ByteSuffix.MiB).to_bytes()),
                300.0,
                "0x20B510DE",
                None): 2}),
        ("sample-system-3", {
            GPU("NVIDIA A30",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(24258, ByteSuffix.MiB).to_bytes()),
                165.0,
                "0x20B710DE",
                None): 8}),
        ("sample-system-5", {
            MIG("NVIDIA A30 MIG-1g.6gb",
                AcceleratorType.Discrete,
                Memory(6, ByteSuffix.GiB),
                165.0,
                "0x20B710DE", None, 1): 32}),
        ("sample-system-6", {
            GPU("NVIDIA A100-SXM4-40GB",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(40536, ByteSuffix.MiB).to_bytes()),
                400.0,
                "0x20B010DE",
                None): 1}),
        ("sample-system-7", {
            GPU("NVIDIA A100-SXM-80GB",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(81251, ByteSuffix.MiB).to_bytes()),
                275.0,
                "0x20B210DE",
                None): 4}),
        ("sample-system-4", {
            GPU("NVIDIA TITAN RTX",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(24220, ByteSuffix.MiB).to_bytes()),
                320.0,
                "0x1E0210DE",
                None): 1,
            GPU("NVIDIA GeForce GT 710",
                AcceleratorType.Discrete,
                Memory.to_1024_base(Memory(2000, ByteSuffix.MiB).to_bytes()),
                None,
                "0x128B10DE",
                None): 1,
        }),
    ]
)
def test_desktop_accelerator_config_detect(spoof_node_id, expected):
    def _test():
        detected = AcceleratorConfiguration.detect()
        assert detected.layout == expected

    def _setup():
        def _override_accelerator_info():
            return get_accelerator_info()
        INFO_SOURCE_REGISTRY.get("nvidia_smi").fn = _override_accelerator_info

        def _override_gpu_info():
            return get_gpu_info(skip_sm_check=True)
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvgpu")).fn = _override_gpu_info

        def _override_mig_info():
            return get_mig_info(skip_sm_check=True)
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvmig")).fn = _override_mig_info

    def _cleanup():
        INFO_SOURCE_REGISTRY.get("nvidia_smi").fn = get_accelerator_info
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvgpu")).fn = get_gpu_info
        INFO_SOURCE_REGISTRY.get(("accelerators", "nvmig")).fn = get_mig_info

    spoof_wrapper(spoof_node_id, [("nvidia_smi",), ("accelerators", "nvgpu"), ("accelerators", "nvmig")], _test, setup=_setup, cleanup=_cleanup)
