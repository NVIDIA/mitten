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
from dataclasses import dataclass
from typing import Any, List, Optional


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
