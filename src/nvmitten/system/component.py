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
from abc import ABC, abstractmethod, abstractclassmethod
from inspect import signature
from typing import Any, Dict, List, Iterable

import dataclasses
import functools
import importlib

from ..interval import NumericRange
from ..json_utils import JSONable


class Component(ABC):
    """Represent some abstract system component. Subclasses of Component should be a dataclass by convention.
    """

    def __init_subclass__(cls):
        # lru_cache call must be here instead of as a decorator for cls.detect for subclass implementations to also be
        # cached.
        cls.detect = functools.lru_cache(maxsize=None)(cls.detect)

    @abstractclassmethod
    def detect(cls) -> Iterable[Component]:
        """Implementations of Component.detect should create and return a list of instances of that Component subclass
        based on information derived from the running system.

        This method is cached with Python's functools.cache to alleviate potentially long system commands that might be
        part of the detection process. As such, `component.detect.cache_clear()` can be used to force the detection
        process to be re-run after the first call.
        """
        raise NotImplementedError

    def summary_description(self) -> Description:
        """Returns a minimal description summarizing this Component instance.
        """
        raise NotImplementedError


class Description(JSONable):
    """Represents a partial description of a system that is more user-readable and easier to define. Used to denote
    known systems in order to match detected hardware.
    """

    def __init__(self,
                 component_cls,
                 _match_ignore_fields: List[str] = None,
                 **kwargs):
        """Creates a description of a Component of type `component_cls`

        Args:
            component_cls: The class of the Component. Must be a dataclass.
            **kwargs: A mapping to denote each field of the component. Argument names should be the same as the
            dataclass fields.
        """
        assert dataclasses.is_dataclass(component_cls), f"Component class {component_cls} must be a dataclass."

        self._component_cls = component_cls

        self._match_ignore_fields = set()
        if _match_ignore_fields:
            self._match_ignore_fields = set(_match_ignore_fields)

        self.mapping = dict()
        for name, f in self.allowed_fields.items():
            if name in kwargs:
                self.mapping[name] = kwargs[name]
            else:
                self.mapping[name] = Any  # Use typing.Any as a filler value to describe a catchall value.

    @functools.cached_property
    def allowed_fields(self) -> Dict[str, dataclasses.Field]:
        return { f.name: f for f in dataclasses.fields(self._component_cls) }

    def matches(self, component: Component) -> bool:
        if not isinstance(component, self._component_cls):
            return False

        # Compare each field
        for name, value in self.mapping.items():
            if name in self._match_ignore_fields:
                continue

            other_val = getattr(component, name)

            if value is Any:
                continue
            elif isinstance(value, NumericRange):
                return value.contains_numeric(other_val)
            elif isinstance(value, Description):
                if not value.matches(other_val):
                    return False
            elif callable(value) and len(signature(value).parameters) == 1:
                if not value(other_val):
                    return False
            elif value != other_val:
                return False

        return True

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Component):
            return self.matches(other)
        elif isinstance(other, Description):
            return self._component_cls == other._component_cls and self.mapping == other.mapping
        else:
            return NotImplemented

    def json_encode(self):
        return {"_component_module": self._component_cls.__module__,
                "_component_classname": self._component_cls.__name__,
                "_match_ignore_fields": list(self._match_ignore_fields),
                "fields": {k: v for k, v in self.mapping.items() if v != Any}}

    @classmethod
    def from_json(cls, d):
        _mod = d["_component_module"]
        _name = d["_component_classname"]

        # Import the class dynamically
        _component_cls = getattr(importlib.import_module(_mod), _name)
        return Description(_component_cls,
                           _match_ignore_fields=d["_match_ignore_fields"],
                           **d["fields"])
