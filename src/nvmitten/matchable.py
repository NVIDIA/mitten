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
from abc import ABC, abstractmethod, abstractclassmethod
from enum import Enum
from typing import Any, Callable, Final, Iterable

import math

from .codestrings import codestringable


class Matchable(ABC):
    """Base class for all Matchables."""

    @abstractmethod
    def matches(self, o: Any) -> bool:
        """Core method for Matchable. This will override the `__eq__` implementation of subclasses of `Matchable`.
        Because of this, implementations should return NotImplemented if classes cannot be compared directly.

        Args:
            o (Any): The object to match against self

        Returns:
            bool: Whether or not `o` and `self` match.
        """
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__eq__ = cls.matches


@codestringable
class MatchAllowList(Matchable):
    """
    Utility class used to match objects against a list of various values.
    """

    def __init__(self, L: Iterable[Any]):
        self.values = list(L)

    def matches(self, o) -> bool:
        """Checks if `o` matches any of the values in the allow list.

        Loops through each value in the allow list and does the following on each:
            1. Perform a LHS equality check
            2. If True, return immediately
            3. Otherwise perform a RHS equality check
            4. If True, return immediately
        If no implementations were found (i.e. all values returned NotImplemented), return NotImplemented.
        Otherwise, return False.
        """
        implementation_found = False
        for v in self.values:
            # Note we want to capture NotImplemented, so we must use __eq__ instead of ==
            lhs_check = v.__eq__(o)
            if lhs_check is True:
                return lhs_check
            elif lhs_check is False:
                implementation_found = True
            else:
                assert lhs_check is NotImplemented

                rhs_check = o.__eq__(v)
                if rhs_check is True:
                    return True
                elif rhs_check is False:
                    implementation_found = True
                else:
                    assert rhs_check is NotImplemented
                    continue

        if implementation_found:
            return False
        else:
            return NotImplemented

    def __hash__(self) -> int:
        return sum(map(hash, self.values))

    def __str__(self) -> str:
        s = "MatchAllowList("
        for v in self.values:
            s += "\n\t" + str(v)
        s += ")"
        return s

    def pretty_string(self) -> str:
        return self.to_codestr()


@codestringable
class MatchAny(Matchable):
    """
    Utility class used to denote any field or object for matching that can match objects (i.e. the field is ignored
    during matching).
    """

    def matches(self, o) -> bool:
        return True

    def __hash__(self) -> int:
        return hash("MatchAny")

    def __str__(self) -> str:
        return "MatchAny()"

    def pretty_string(self) -> str:
        return self.to_codestr()

    def to_codestr(self) -> str:
        return "MATCH_ANY"


MATCH_ANY: Final[Matchable] = MatchAny()


class MatchFloatApproximate(Matchable):
    """
    Utility class to compare 2 matchables that represent floating point numbers with an approximation.
    """

    def __init__(self, o: Matchable, to_float_fn: Callable[Matchable, float], rel_tol: float = 0.05):
        """Creates a MatchFloatApproximate given the base Matchable and a function to return a float representation of
        the Matchable.

        Args:
            o (Matchable): The object to wrap around
            to_float_fn (Callable[Matchable, float]): Function that takes a Matchable and returns a floating point
                                                      representation. The input parameter should accept the same type as
                                                      `o`.
            rel_tol (float): The relative tolerance to use for the float comparison. (Default: 0.05)
        """
        self.o = o
        self.to_float_fn = to_float_fn
        self.rel_tol = rel_tol

    def matches(self, other) -> bool:
        if self.o.__class__ == other.__class__:
            return math.isclose(self.to_float_fn(self.o), self.to_float_fn(other), rel_tol=self.rel_tol)
        elif self.__class__ == other.__class__:
            return self.o == other.o
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.o) + hash('MatchFloatApproximate')

    def __str__(self) -> str:
        return f"MatchFloatApproximate(value={self.o}, rel_tol={self.rel_tol})"

    def pretty_string(self) -> str:
        return f"approx. {self.o.quantity} {self.o.byte_suffix.name}"


class MatchNumericThreshold(Matchable):
    """
    Utility class to compare 2 matchables that represent numeric values, using some value as a threshold as either the
    min or max threshold.
    """

    def __init__(self, o: Matchable, to_numeric_fn: Callable[Matchable, Number], min_threshold: bool = True):
        """
        Creates a MatchNumericThreshold given the base Matchable, a function to return a Numeric representation of the
        Matchable, and whether or not the base Matchable is the minimum threshold or maximum threshold.

        Args:
            o (Matchable): The object to wrap around
            to_numeric_fn (Callable[Matchable, Number]): Function that takes a Matchable and returns a Number. The input
                                                         parameter should accept the same type as `o`.
            min_threshold (bool): If True, uses `o` as the minimum threshold when comparing, so that `self.matches`
                                  returns True if `other` is larger than `o`. Otherwise, uses `o` as the max threshold,
                                  so `self.matches` returns True if `other` is smaller than `o`. (Default: True)
        """
        self.o = o
        self.to_numeric_fn = to_numeric_fn
        self.min_threshold = min_threshold
        self.compare_symbol = ">=" if self.min_threshold else "<="

    def matches(self, other) -> bool:
        if self.o.__class__ == other.__class__:
            if self.min_threshold:
                return self.to_numeric_fn(other) >= self.to_numeric_fn(self.o)
            else:
                return self.to_numeric_fn(other) <= self.to_numeric_fn(self.o)
        elif self.__class__ == other.__class__:
            return self.o == other.o and self.min_threshold == other.min_threshold
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.o) + hash('MatchNumericThreshold')

    def __str__(self) -> str:
        return f"MatchNumericThreshold({self.compare_symbol}  {self.o})"

    def pretty_string(self) -> str:
        return f"{self.compare_symbol} {self.o.pretty_string()}"


class MatchableEnum(Enum):
    """Meant to be used as a parent class for any Enum that contains only Matchables.
    """

    @classmethod
    def get_first_match(cls, x) -> Matchable:
        """Returns the first Enum member that matches x."""
        for elem in cls:
            if elem.value == x:
                return elem
        return None
