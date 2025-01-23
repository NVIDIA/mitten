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
from dataclasses import dataclass
from typing import Any, List, Optional

import math

from .json_utils import JSONable


@dataclass(eq=True, frozen=True)
class Interval:
    """Defines an inclusive range of integers [a, b]. i.e. Interval[1, 3] is the same as {1, 2, 3}.
    """
    start: int
    end: Optional[int] = None

    def __post_init__(self):
        if self.end is None:
            object.__setattr__(self, 'end', self.start)

        if self.start > self.end:
            tmp = self.start
            object.__setattr__(self, 'start', self.end)
            object.__setattr__(self, 'end', tmp)

    def intersects(self, o: Any):
        """Returns whether or not another interval overlaps with self. This includes trivial overlaps where the start of
        one interval equals the end of the other, since Intervals are inclusive.

        Args:
            o (Any) - An arbitrary object.

        Returns:
            bool - If o is not an Interval, returns NotImplemented. Otherwise, returns whether or not the intervals
                   overlap.
        """
        if o.__class__ != Interval:
            return NotImplemented
        return (min(self.end, o.end) - max(self.start, o.start)) >= 0

    def __str__(self):
        if self.start == self.end:
            return str(self.start)
        else:
            return f"{self.start}-{self.end}"

    def __iter__(self):
        return range(self.start, self.end + 1)

    def to_list(self):
        return [i for i in range(self.start, self.end + 1)]

    def to_set(self):
        return set(self.to_list())

    @classmethod
    def from_str(cls, s: str):
        toks = s.split("-")
        if len(toks) > 2 or len(toks) == 0:
            raise ValueError(f"Cannot convert string '{s}' to {cls.__name__}")
        elif len(toks) == 1:
            return cls(int(toks[0]))
        else:
            return cls(int(toks[0]), int(toks[1]))

    @classmethod
    def build_interval_list(cls, nums: List[int]) -> List[Interval]:
        """Returns a list of Intervals representing the numbers in `nums`.

        Args:
            nums (List[int]): A list of integers to turn into a list of Intervals

        Returns:
            List[Interval]: An list of Intervals that is equivalent to the input list.
        """
        if len(nums) == 0:
            return []

        nums = list(sorted(set(nums)))
        intervals = [(nums[0], nums[0])]
        for x in nums[1:]:
            if x == intervals[-1][1] + 1:
                intervals[-1] = (intervals[-1][0], x)
            else:
                intervals.append((x, x))
        return [cls(*t) for t in intervals]

    @classmethod
    def interval_list_from_str(cls, s: str) -> List[Interval]:
        L = []
        for tok in s.split(","):
            L.append(cls.from_str(tok))
        return L


class NumericRange(JSONable):
    def __init__(self,
                 start: float,
                 end: Optional[float] = None,
                 rel_tol: Optional[float] = None,
                 abs_tol: Optional[float] = None):
        """Creates a NumericRange.

        Args:
            start (float): The start of the numeric range. If end is None, this is treated as the base value for either
                           relative or absolute tolerance.
            end (float): The end of the numeric range. If specified, must be greater than `start`. If unspecified,
                         exactly 1 of `rel_tol` and `abs_tol` must not be None. (Default: None)
            rel_tol (float): Relative tolerance. Ignored if end is not None. (Default: None)
            abs_tol (float): Absolute tolerance. Ignored if end is not None or rel_tol is not None. (Default: None)
        """
        if end is not None:
            assert start <= end, f"Range end {end} must be at least start {start}"
            self.start = start
            self.end = end
        elif rel_tol is not None:
            assert 0.0 <= rel_tol and rel_tol <= 1.0, f"Relative tolerance must be between 0 and 1, got {rel_tol}"
            self.start = start * (1.0 - rel_tol)
            self.end = start * (1.0 + rel_tol)

            self._base = start
            self._rel_tol = rel_tol
        elif abs_tol is not None:
            assert abs_tol >= 0, f"Absolute tolerance must be non-negative, got {abs_tol}"
            self.start = start - abs_tol
            self.end = start + abs_tol

            self._base = start
            self._abs_tol = abs_tol
        else:
            self.start = start
            self.end = math.inf

    def contains_range(self, other: NumericRange) -> bool:
        return self.start <= other.start and other.end <= self.end

    def contains_numeric(self, other: float) -> bool:
        return self.start <= other and other <= self.end

    def __eq__(self, o: Any) -> bool:
        if not isinstance(o, NumericRange):
            return NotImplemented
        return self.start == o.start and self.end == o.end

    def __str__(self):
        if hasattr(self, "_abs_tol"):
            return f"NumericRange(mode=Absolute, {self._base} plus/minus {self._abs_tol} [{self.start}, {self.end}])"
        elif hasattr(self, "_rel_tol"):
            return f"NumericRange(mode=Relative, {self._base} ~ {self._rel_tol} [{self.start}, {self.end}])"
        else:
            return f"NumericRange([{self.start}, {self.end}])"

    def json_encode(self):
        if hasattr(self, "_abs_tol"):
            return {"start": self._base,
                    "abs_tol": self._abs_tol}
        elif hasattr(self, "_rel_tol"):
            return {"start": self._base,
                    "rel_tol": self._rel_tol}
        else:
            return {"start": self.start,
                    "end": self.end}

    @classmethod
    def from_json(cls, d):
        return NumericRange(**d)
