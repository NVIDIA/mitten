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
