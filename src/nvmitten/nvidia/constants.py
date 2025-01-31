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

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from enum import Enum, unique

import tensorrt as trt

from ..aliased_name import AliasedName, AliasedNameEnum
from ..json_utils import JSONable
from ..mlcommons.inference.constants import AccuracyTarget, Benchmark


TRT_LOGGER: Final[trt.Logger] = trt.Logger(trt.Logger.INFO)


@dataclass(frozen=True)
class ComputeSM(JSONable):
    major: int
    minor: int

    def __post_init__(self):
        # SM numbers have value assertions
        assert self.major >= 0
        assert self.minor >= 0
        assert self.minor < 10

    def __int__(self):
        return 10 * self.major + self.minor

    @classmethod
    def from_int(cls, i: int) -> ComputeSM:
        """Creates a ComputeSM from an int value.

        Args:
            i (int): A positive SM value. Negative values will raise ValueError.

        Returns:
            ComputeSM
        """
        if i < 0:
            raise ValueError(f"Negative value {i} is not a valid compute SM")
        major = i // 10
        minor = i % 10
        return ComputeSM(major, minor)

    def json_encode(self):
        return int(self)

    @classmethod
    def from_json(cls, d):
        return cls.from_int(d)


@unique
class Action(AliasedNameEnum):
    """Names of actions performed by our MLPerf Inference pipeline."""

    GenerateConfFiles: AliasedName = AliasedName("generate_conf_files")
    GenerateEngines: AliasedName = AliasedName("generate_engines")
    Calibrate: AliasedName = AliasedName("calibrate")
    RunHarness: AliasedName = AliasedName("run_harness")
    RunAuditHarness: AliasedName = AliasedName("run_audit_harness")


@unique
class HarnessType(AliasedNameEnum):
    """Possible harnesses a benchmark can use."""
    LWIS: AliasedName = AliasedName("lwis")
    Custom: AliasedName = AliasedName("custom")
    Triton: AliasedName = AliasedName("triton")
    HeteroMIG: AliasedName = AliasedName("hetero")
    # TODO: once triton harness unification is completed, TritonUnified will replace Triton
    TritonUnified: AliasedName = AliasedName("triton_unified")


@unique
class PowerSetting(AliasedNameEnum):
    """Possible power settings the system can be set in when running a benchmark."""
    MaxP: AliasedName = AliasedName("MaxP")
    MaxQ: AliasedName = AliasedName("MaxQ")


@dataclass(frozen=True)
class WorkloadSetting:
    """
    Describes the various settings used when running a benchmark workload. These are usually for different use cases
    that MLPerf Inference allows (i.e. power submission), or running the same workload with different software (i.e.
    Triton).
    """
    harness_type: HarnessType = HarnessType.Custom
    """HarnessType: Harness to use for this workload. Default: HarnessType.Custom"""

    accuracy_target: AccuracyTarget = AccuracyTarget.k_99
    """AccuracyTarget: Accuracy target for the benchmark. Default: AccuracyTarget.k_99"""

    power_setting: PowerSetting = PowerSetting.MaxP
    """PowerSetting: Power setting for the system during this workload. Default: PowerSetting.MaxP"""

    def __str__(self) -> str:
        return f"WorkloadSetting({self.harness_type}, {self.accuracy_target}, {self.power_setting})"

    def shortname(self) -> str:
        return f"{self.harness_type.value.name}_{self.accuracy_target.name}_{self.power_setting.value.name}"

    def as_dict(self) -> Dict[str, Any]:
        """
        Convenience wrapper around dataclasses.asdict to convert this WorkloadSetting to a dict().

        Returns:
            Dict[str, Any]: This WorkloadSetting as a dict
        """
        return asdict(self)


G_DEFAULT_HARNESS_TYPES: Dict[Benchmark, HarnessType] = {
    Benchmark.BERT: HarnessType.Custom,
    Benchmark.DLRMv2: HarnessType.Custom,
    Benchmark.ResNet50: HarnessType.LWIS,
    Benchmark.Retinanet: HarnessType.LWIS,
    Benchmark.UNET3D: HarnessType.Custom,
}
"""Dict[Benchmark, HarnessType]: Defines the default harnesses (non-Triton) used for each benchmark."""


def config_ver_to_workload_setting(benchmark: Benchmark, config_ver: str) -> WorkloadSetting:
    """This method is a temporary workaround to retain legacy behavior as the codebase is incrementally refactored to
    use the new Python-style BenchmarkConfiguration instead of the old config.json files.

    Converts a legacy 'config_ver' ID to a new-style WorkloadSetting.

    Args:
        benchmark (Benchmark):
            The benchmark that is being processed. Used to decide the HarnessType.
        config_ver (str):
            The old-style 'config_ver' ID

    Returns:
        WorkloadSetting: The equivalent WorkloadSetting for the benchmark/config_ver.
    """
    harness_type = G_DEFAULT_HARNESS_TYPES[benchmark]
    if "triton_unified" in config_ver:
        harness_type = HarnessType.TritonUnified
    elif "openvino" in config_ver or "triton" in config_ver:
        harness_type = HarnessType.Triton
    elif "hetero" in config_ver:
        harness_type = HarnessType.HeteroMIG

    accuracy_target = AccuracyTarget.k_99
    if "high_accuracy" in config_ver:
        accuracy_target = AccuracyTarget.k_99_9

    power_setting = PowerSetting.MaxP
    if "maxq" in config_ver:
        power_setting = PowerSetting.MaxQ

    return WorkloadSetting(harness_type=harness_type, accuracy_target=accuracy_target, power_setting=power_setting)
