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
from abc import ABC, abstractmethod, abstractclassmethod
from enum import Enum
from typing import Callable, Final

import dataclasses
import math

from ..aliased_name import AliasedName
from ..matchable import Matchable


class Hardware(Matchable):
    """
    Abstract class for representing hardware, such as a CPU or GPU, that can be matched with other hardware
    components and be detected programmatically. Subclasses of Hardware should be dataclasses by convention.
    """

    def identifiers(self):
        """Returns the identifiers used for the match behavior (__eq__) and __hash__.

        Returns:
            Tuple[Any...]: A tuple of identifiers
        """
        raise NotImplementedError

    @classmethod
    def detect(cls) -> Hardware:
        raise NotImplementedError

    def matches(self, other) -> bool:
        """Matches this Hardware component with 'other'.

        If other is the same class as self, compare the identifiers.
        If the other object is a MatchAny or MatchAllowList, it will use other's .matches() instead.
        Returns False otherwise.

        Returns:
            bool: Equality based on the rules described above
        """
        if other.__class__ == self.__class__:
            return self.identifiers() == other.identifiers()
        return NotImplemented
