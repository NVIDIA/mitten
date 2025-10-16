# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import dataclasses as dcls
import enum
import os
from pathlib import Path
import inspect
from typing import Any, Callable, Dict, List, Optional


class AutoConfStrategy(enum.Enum):
    """Strategy for applying configuration values to fields."""
    Replace = "replace"
    """Replace the entire value"""
    DictUpdate = "dict_update"
    """Merge dict values using dict.update()"""


@dcls.dataclass(frozen=True)
class Field:
    """Represents a configurable parameter.
    """

    name: str
    """str: Unique identifier for this field"""

    description: str = ""
    """str: Description of this field's meaning and usage. Used for argparse.ArgumentParser."""

    from_string: Callable[[str], Any] = None
    """Callable[[str], Any]: Single parameter function that takes in a string and returns a valid value to use for this
    field. Used to convert the string parsed from CLI flags to a usable value."""

    disallow_default: bool = False
    """bool: If True, default values will not be used for this Field, and this Field must be manually specified via
    either a Configuration or parsed from argparse.ArgumentParser."""

    disallow_argparse: bool = False
    """bool: If True, this Field will not be parsed from argparse.ArgumentParser."""

    argparse_opts: Optional[Dict[str, Any]] = None
    """Dict[str, Any]: Optional kwargs to use for Argparse.Argument"""

    from_environ: Optional[str] = None
    """str: If set, check this environment variable for value instead of using CLI flags. Implicitly sets
    disallow_argparse to True."""

    autoconf_strategy: AutoConfStrategy = AutoConfStrategy.Replace
    """AutoConfStrategy: Strategy for applying configuration values. Replace (default) replaces the entire value,
    DictUpdate merges dict values using dict.update()."""

    _created_in: os.PathLike = dcls.field(init=False)
    """os.PathLike: Fields are meant to be singletons and can be organized by the files they are defined in. This
    attribute stores where this Field was first initialized.
    """

    def __post_init__(self):
        if self.from_environ:
            object.__setattr__(self, 'disallow_argparse', True)
        # Get the filename of the file that created this Field
        # 0 - __post_init__
        # 1 - __init__
        # 2 - caller
        object.__setattr__(self, '_created_in', Path(inspect.stack()[2].filename))

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        # Only allow equality for exact object matches to prevent duplicate fields with different bindings.
        return self is other

    def __copy__(self):
        # Since we are matching singleton instances, __copy__ should be a no-op
        return self

    def __deepcopy__(self, memo):
        # Like __copy__, __deepcopy__ should also be a no-op
        return self

    def add_to_argparse(self, argparser: argparse.ArgumentParser):
        if self.disallow_argparse:
            return

        if self.argparse_opts:
            opts = dict(self.argparse_opts)
        else:
            opts = dict()
        opts["help"] = self.description
        if self.from_string is bool:
            opts["action"] = "store_true"
        elif self.from_string:
            opts["type"] = self.from_string

        argparser.add_argument(f"--{self.name}", **opts)


def parse_fields(fields: List[Field], *args, **kwargs):
    d = dict()

    parser = argparse.ArgumentParser(allow_abbrev=False,
                                     argument_default=argparse.SUPPRESS,
                                     add_help=False)
    for f in fields:
        if f.from_environ:
            if f.from_environ in os.environ:
                _v = os.environ[f.from_environ]
                if f.from_string:
                    _v = f.from_string(_v)
                d[f.name] = _v
        else:
            f.add_to_argparse(parser)

    d.update(vars(parser.parse_known_args(*args, **kwargs)[0]))
    return d
