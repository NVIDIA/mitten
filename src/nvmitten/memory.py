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
from functools import total_ordering
from typing import Any, Dict, Final, Optional, Union, Tuple

import re
import math

from .constants import QUANTITY_UNIT_FORMAT, ByteSuffix
from .json_utils import JSONable


@total_ordering
@dataclass(eq=True, frozen=True)
class Memory(JSONable):
    """Represents an amount of computer memory."""

    quantity: float
    """float: Amount of memory units"""

    byte_suffix: ByteSuffix
    """ByteSuffix: The unit of memory to use"""

    _num_bytes: int = field(init=False)

    def __post_init__(self):
        num_bytes = self.quantity * self.byte_suffix.to_bytes()
        # This is a float, and potentially may look like xyz.0, xyz.99999998, xyz.0000518, or xyz.1abc.
        # The last case is not valid, but the second and third cases are floating point imprecision.
        # However, this is only in the case where self.byte_suffix is ByteSuffix.Byte or has a metric prefix, since
        # multiplying a float by 1024 is unlikely to produce an integer, ie. 1.5 KiB -> 1.5 * 1024 B is an integer, but
        # 3.14 KiB -> 3.14 * 1024 is not.
        dec = num_bytes - math.trunc(num_bytes)
        # This is the value after the decimal point. If there is floating point imprecision, we don't care because we
        # check with math.isclose. Also note we check math.isclose between 1+dec and 1, since rel_tol would fail in the
        # happy case where dec = 0.
        if self.byte_suffix.value[0] in (1, 1000) and not (
                math.isclose(1 + dec, 1, rel_tol=1e-3) or
                math.isclose(1, dec, rel_tol=1e-3)):
            raise ValueError(f"Memory({self.quantity}, {self.byte_suffix}) converts to {num_bytes} bytes, "
                             "which is not an integer")
        # Note we cannot use '=' since this is a frozen dataclass. This is the official solution that dataclass uses:
        object.__setattr__(self, '_num_bytes', round(num_bytes))

    @classmethod
    def from_string(cls, s):
        """Creates a Memory object from a string, formatted like '[float][optional space][byte suffix]'
        ByteSuffix MUST be formatted in capitalized letters except for -ibibyte values. i.e. 'GB' and 'GiB' will be
        accepted, but 'gb' will not.
        """
        m = QUANTITY_UNIT_FORMAT.fullmatch(s)
        if m is None:
            raise ValueError(f"Cannot convert string '{s}' to a Memory object. Invalid format.")
        suffix = m.group(2)
        if suffix not in ByteSuffix._member_map_:
            raise ValueError(f"Cannot convert string '{s}' to a Memory object. Invalid suffix.")
        quant = float(m.group(1))
        if quant < 0:
            raise ValueError(f"Cannot convert string '{s}' to a Memory object. Negative quantity.")
        return Memory(quant, ByteSuffix[suffix])

    def convert(self, byte_suffix):
        """Converts a memory representation to an equivalent memory representation with a different ByteSuffix unit
        (i.e. from MB to GB), maintaining the same number of bytes."""
        return Memory._to_base(byte_suffix, self._num_bytes)

    def __eq__(self, o: Any) -> bool:
        if o.__class__ is not Memory:
            return NotImplemented
        return self._num_bytes == o._num_bytes

    def __lt__(self, o: Any) -> bool:
        if o.__class__ is not Memory:
            return NotImplemented
        return self._num_bytes < o._num_bytes

    def __add__(self, o: Any) -> Memory:
        if isinstance(o, Memory):
            offset = o._num_bytes
        else:
            offset = self.byte_suffix.to_bytes() * o

        base = self.byte_suffix.value[0]
        return Memory._to_base(base, self._num_bytes + offset)

    def __sub__(self, o: Any) -> Memory:
        if isinstance(o, Memory):
            offset = o._num_bytes
        else:
            offset = self.byte_suffix.to_bytes() * o

        base = self.byte_suffix.value[0]
        return Memory._to_base(base, self._num_bytes - offset)

    def __mul__(self, o: Any) -> Memory:
        if isinstance(o, Memory):
            raise TypeError("Multiplying 2 Memory objects is ambiguous. Only float scale factors are supported.")
        base = self.byte_suffix.value[0]
        return Memory._to_base(base, self._num_bytes * o)

    def __truediv__(self, o: Any) -> Memory:
        if isinstance(o, Memory):
            raise TypeError("Dividing 2 Memory objects is ambiguous. Only float scale factors are supported.")
        base = self.byte_suffix.value[0]
        return Memory._to_base(base, self._num_bytes / o)

    @classmethod
    def _to_base(cls, base, n):
        max_exp = float("+inf")
        if type(base) is ByteSuffix:
            base, max_exp = base.value

        if n < base:
            return Memory(n, ByteSuffix.B)

        exp = 0
        while n >= base and exp < max_exp:
            exp += 1
            n /= base
        return Memory(n, ByteSuffix((base, exp)))

    @classmethod
    def to_1000_base(cls, n):
        return cls._to_base(1000, n)

    @classmethod
    def to_1024_base(cls, n):
        return cls._to_base(1024, n)

    def to_bytes(self):
        return self._num_bytes

    def pretty_string(self) -> str:
        """Formatted, human-readable string displaying the data in the Memory

        Returns:
            str: 'Pretty-print' string representation of the Memory
        """
        return f"{self.quantity:.2f} {self.byte_suffix.name}"

    def json_encode(self):
        return {"num_bytes": self._num_bytes,
                "base": self.byte_suffix.value[0],
                "human_readable_UNUSED": self.pretty_string()}

    @classmethod
    def from_json(cls, d):
        return Memory._to_base(d["base"], d["num_bytes"])
