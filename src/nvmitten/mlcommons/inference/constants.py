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
from enum import Enum, unique

from ...aliased_name import AliasedName, AliasedNameEnum


__doc__ = """Stores constants and Enums related to MLPerf Inference"""


@unique
class Benchmark(AliasedNameEnum):
    """Names of supported Benchmarks in MLPerf Inference."""

    BERT: AliasedName = AliasedName("bert")
    GPTJ: AliasedName = AliasedName("gptj", ("gptj6b", "gpt-j"))
    LLAMA2: AliasedName = AliasedName("llama2-70b", ("llama2", "llama-v2", "llama-v2-70b"))
    DLRMv2: AliasedName = AliasedName("dlrm-v2", ("dlrm_v2", "dlrmv2", "dcnv2", "dlrm_dcn"))
    RNNT: AliasedName = AliasedName("rnnt", ("rnn-t",))
    ResNet50: AliasedName = AliasedName("resnet50", ("resnet",))
    UNET3D: AliasedName = AliasedName("3d-unet", ("3dunet", "unet", "3d-unet-kits", "3d-unet-kits19"))
    Retinanet: AliasedName = AliasedName("retinanet", ("ssd-retinanet", "resnext", "ssd-resnext"))
    SDXL: AliasedName = AliasedName("sdxl", ("sdxl-base", "diffusion", "stable-diffusion", "stable-diffusion-xl"))


@unique
class Scenario(AliasedNameEnum):
    """Names of supported workload scenarios in MLPerf Inference.

    IMPORTANT: This is **not** interchangeable or compatible with the TestScenario Enum from the mlperf_loadgen Python
    bindings. Make sure that any direct calls to mlperf_loadgen use the TestScenario Enum from loadgen, **not** this
    Enum.
    """

    Offline: AliasedName = AliasedName("Offline")
    Server: AliasedName = AliasedName("Server")
    SingleStream: AliasedName = AliasedName("SingleStream", ("single-stream", "single_stream"))
    MultiStream: AliasedName = AliasedName("MultiStream", ("multi-stream", "multi_stream"))


@unique
class AuditTest(AliasedNameEnum):
    """Audit test names"""

    TEST01: AliasedName = AliasedName("TEST01")
    TEST04: AliasedName = AliasedName("TEST04")
    TEST05: AliasedName = AliasedName("TEST05")
    TEST06: AliasedName = AliasedName("TEST06")


@unique
class AccuracyTarget(Enum):
    """Possible accuracy targets a benchmark must meet. Determined by MLPerf Inference committee."""
    k_99: float = .99
    k_99_9: float = .999


G_HIGH_ACC_ENABLED_BENCHMARKS: Tuple[Benchmark, ...] = (
    Benchmark.BERT,
    Benchmark.DLRMv2,
    Benchmark.GPTJ,
    Benchmark.LLAMA2,
    Benchmark.UNET3D,
)
"""Tuple[Benchmark, ...]: Benchmarks that have 99.9% accuracy targets"""


G_DATACENTER_BENCHMARKS: Tuple[Benchmark, ...] = (
    Benchmark.BERT,
    Benchmark.GPTJ,
    Benchmark.LLAMA2,
    Benchmark.DLRMv2,
    Benchmark.RNNT,
    Benchmark.ResNet50,
    Benchmark.Retinanet,
    Benchmark.UNET3D,
    Benchmark.SDXL,
)
"""Tuple[Benchmark, ...]: Benchmarks for the Datacenter submission category"""


G_EDGE_BENCHMARKS: Tuple[Benchmark, ...] = (
    Benchmark.BERT,
    Benchmark.RNNT,
    Benchmark.ResNet50,
    Benchmark.Retinanet,
    Benchmark.UNET3D,
    Benchmark.SDXL,
)
"""Tuple[Benchmark, ...]: Benchmarks for the Edge submission category"""
