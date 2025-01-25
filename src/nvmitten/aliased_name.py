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
from typing import Any, Dict, Final, Optional, Union, Tuple

import re


@dataclass(eq=False, frozen=True)
class AliasedName:
    """
    Represents a name that has given aliases that are considered equivalent to the original name.
    """

    name: Optional[str]
    """Optional[str]: The main name this AliasedName is referred to as. None is reserved as a special value."""

    aliases: Tuple[str, ...] = tuple()
    """Tuple[str, ...]: A tuple of aliases that are considered equivalent, and should map to the main name."""

    patterns: Tuple[re.Pattern, ...] = tuple()
    """Tuple[re.Pattern, ...]: A tuple of regex patterns this AliasedName can match with. These have lower precendence
    than aliases and will only be checked if aliases has been exhausted without a match. Patterns are only used if
    the object to match is an instance of str."""

    def __str__(self):
        return self.name

    def __hash__(self):
        # Needs to be implemented since we specify eq=False. See Python dataclass documentation.
        if self.name is None:
            return hash(None)
        return hash(self.name.lower())

    def __add__(self, other):
        if other.__class__ is AliasedName:
            raise Exception("Concatenating AliasedNames is ambiguous. One parameter should be a str.")
        elif other.__class__ is str:
            if len(self.patterns) > 0:
                raise Exception("Cannot concatenate AliasedName with regex patterns.")
            return AliasedName(self.name + other, [alias + other for alias in self.aliases])
        return NotImplemented

    def __eq__(self, other: Union[AliasedName, str, None]) -> bool:
        """
        Case insensitive equality check. Can be compared with another AliasedName or a str.

        If other is an AliasedName, returns True if self.name is case-insensitive equivalent to other.name.

        If other is a str, returns True if other is case-insensitive equivalent to self.name or any of the elements of
        self.aliases, or if it is a full match of any regex patterns in self.patterns.

        Args:
            other (Union[AliasedName, str, None]): The object to compare to

        Returns:
            bool: True if other is considered equal by the above rules. False otherwise, or if other is of an
            unrecognized type.
        """

        if other is None:
            return self.name is None

        if isinstance(other, AliasedName):
            if self.name is None or other.name is None:
                return self.name == other.name
            return self.name.lower() == other.name.lower()
        elif isinstance(other, str):
            if self.name is None:
                return False
            other_lower = other.lower()
            if self.name.lower() == other_lower or other_lower in (x.lower() for x in self.aliases):
                return True

            # No aliases matched, iterate through each regex.
            for pattern in self.patterns:
                if pattern.fullmatch(other):
                    return True
            return False
        else:
            return NotImplemented


class AliasedNameEnum(Enum):
    """Used as a parent class for any Enum that has exclusively AliasedName values.
    """

    @classmethod
    def as_aliased_names(cls) -> List[AliasedName]:
        return [elem.value for elem in cls]

    @classmethod
    def as_strings(cls) -> List[str]:
        return list(map(str, cls.as_aliased_names()))

    @classmethod
    def get_match(cls, name: Union[AliasedName, str]) -> Optional[AliasedNameEnum]:
        """
        Attempts to return the element of this enum that is equivalent to `name`.

        Args:
            name (Union[AliasedName, str]):
                The name of an element we want

        Returns:
            Optional[AliasedName]: The AliasedName if found, None otherwise
        """
        for elem in cls:
            if elem.value == name or elem == name:
                return elem
        return None

    def __eq__(self, other: Any) -> bool:
        """
        __eq__ override for members. Will compare directly if `other` is of the same __class__. Otherwise will attempt
        to use the __eq__ of the value.

        Args:
            other (Any):
                The object to compare to

        Returns:
            bool: True if other is equivalent to self directly, or self.value. False otherwise.
        """
        if self.__class__ is other.__class__:
            return self is other
        else:
            return self.value == other

    def __hash__(self) -> int:
        return hash(self.value)

    def valstr(self) -> str:
        """
        Convenience method to get the string representation of this Enum member's value.

        Returns:
            str: self.value.__str__()
        """
        return str(self.value)
