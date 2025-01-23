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
from enum import Enum, unique
from typing import Any, Dict, Final, Optional, Union, Tuple

import os
import re
import math
import importlib
import importlib.util

from .aliased_name import AliasedName, AliasedNameEnum


__doc__ = """Stores constants and Enums related to MLPerf Inference"""


QUANTITY_UNIT_FORMAT = re.compile(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)) ?(\S+)")
"""re.Pattern: Regex to parse strings that indicate numeric quantities with a unit. Derived from
               https://docs.python.org/3/library/re.html#simulating-scanf

               Disallows scientific notations (ie. 1e5).

               Will match:
                    - "12.5 KB"
                    - "15TB"
                    - "14 lbs"
                    - "0.2cm"
                    - "-19.7KB"     This will still match even though "negative memory" isn't valid
               Will not match:
                    - "1e6 GB"      Reason: Uses scientific notation
                    - "4  KB"       Reason: Too many spaces between quantity and unit
                    - "22"          Reason: No unit string

               match(1) - unitless, numeric quantity of memory
               match(2) - the byte suffix, denoting the unit
"""


@unique
class ByteSuffix(Enum):
    B = (1, 0)

    KiB = (1024, 1)
    MiB = (1024, 2)
    GiB = (1024, 3)
    TiB = (1024, 4)

    KB = (1000, 1)
    MB = (1000, 2)
    GB = (1000, 3)
    TB = (1000, 4)

    def to_bytes(self):
        base, exponent = self.value
        return base ** exponent


@unique
class Precision(AliasedNameEnum):
    """Different numeric precisions that can be used by benchmarks. Not all benchmarks can use all precisions."""

    NVFP4: AliasedName = AliasedName("nvfp4")
    FP8: AliasedName = AliasedName("fp8")
    INT8: AliasedName = AliasedName("int8")
    FP16: AliasedName = AliasedName("fp16")
    FP32: AliasedName = AliasedName("fp32")


@unique
class InputFormats(AliasedNameEnum):
    """Different input formats that can be used by benchmarks. Not all benchmarks can use all input formats."""
    Linear: AliasedName = AliasedName("linear")
    CHW4: AliasedName = AliasedName("chw4")
    DHWC8: AliasedName = AliasedName("dhwc8")
    CDHW32: AliasedName = AliasedName("cdhw32")


@unique
class CPUArchitecture(AliasedNameEnum):
    """Various CPU Architectures"""
    x86_64: AliasedName = AliasedName("x86_64")
    aarch64: AliasedName = AliasedName("aarch64")
