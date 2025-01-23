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

from enum import Enum
from numbers import Number
from types import ModuleType
from typing import Any

import dataclasses

from .aliased_name import AliasedName


def codestringable(klass: ModuleType) -> ModuleType:
    """Given a class, adds a mechanism to save the initializer arguments, allowing the instance to be reproduced.
    Note that this cannot be a MixinClass or parent class, as doing so would require any child class to call
    super().__init__, but this is up to the user and could cause bugs by not passing in the correct arguments and such.

    Using an `@codestringable` decorator appears to be the least intrusive way of enabling this feature without some
    heavy usage of inspect and source parsing.

    Args:
        klass (ModuleType): A class to make codestringable

    Returns:
        ModuleType: `klass` after its init method has been modified
    """
    klass._base_init = klass.__init__

    def _new_init(ref, *args, **kwargs):
        ref._init_args = args
        ref._init_kwargs = kwargs
        klass._base_init(ref, *args, **kwargs)
    klass.__init__ = _new_init

    if not hasattr(klass, "to_codestr"):
        def _to_codestr(ref):
            arg_strs = list(map(obj_to_codestr, ref._init_args))
            for k, v in ref._init_kwargs.items():
                arg_strs.append(f"{k}={obj_to_codestr(v)}")
            arg_str = ", ".join(arg_strs)
            return f"{ref.__class__.__name__}({arg_str})"
        klass.to_codestr = _to_codestr

    return klass


def obj_to_codestr(o: Any) -> str:
    """Returns a str representing a code object.

    Args:
        o (Any): The object to convert

    Returns:
        str: A string representing the code used to create the object `o`
    """
    if o is None:
        return "None"
    elif hasattr(o, "to_codestr") and callable(o.to_codestr):
        return o.to_codestr()
    elif isinstance(o, Enum):
        return str(o)
    elif isinstance(o, str) or isinstance(o, AliasedName):
        return f"\"{str(o)}\""
    elif isinstance(o, Number) or isinstance(o, bool):
        return str(o)
    elif isinstance(o, list):
        codestr = ", ".join(map(obj_to_codestr, o))
        return f"[{codestr}]"
    elif isinstance(o, dict):
        values = []
        for k, v in o.items():
            values.append((obj_to_codestr(k), obj_to_codestr(v)))
        values = [f"{t[0]}: {t[1]}" for t in values]
        codestr = ", ".join(values)
        return f"{{{codestr}}}"
    elif dataclasses.is_dataclass(o):
        class_name = o.__class__.__name__
        s = f"{class_name}("
        # Add each field as a new line
        L = []
        for f in dataclasses.fields(o):
            if not f.init:
                continue
            field_name = f.name
            field_str = obj_to_codestr(getattr(o, field_name))
            L.append(f"{field_name}={field_str}")
        s += ", ".join(L)
        s += ")"
        return s
    raise ValueError(f"Cannot convert object of type {type(o)} to code-string.")
