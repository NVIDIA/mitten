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
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, Optional, Tuple

import inspect
import time
import traceback

from .errors import MissingParentOutputKey
from ..debug import DebugManager, DebuggableMixin


@unique
class OperationStatus(Enum):
    PASSED = 0
    SKIPPED = 1
    INTERRUPTED = 2
    FAILED = 3


@dataclass(frozen=True)
class OperationResult:
    """Represents the result of an Operation. This is constructed by the Pipeline.
    """
    status: OperationStatus
    value: dict
    start_time_ns: int
    end_time_ns: int = field(init=False)
    exception: Exception = None
    trace: traceback.TracebackException = None

    def __post_init__(self):
        object.__setattr__(self, "end_time_ns", time.time_ns())
        assert self.end_time_ns > self.start_time_ns, "start_time_ns cannot be in the future"

    def duration(unit: str = 'ns') -> float:
        delta = self.end_time_ns - self.start_time_ns
        scales = {"ns": 1,
                  "us": 1000,
                  "ms": 1000 ** 2,
                  "s": 1000 ** 3}
        if unit not in scales:
            raise RuntimeError(f"OperationResult.duration received invalid unit: {unit}")
        return delta / scales[unit]

    def __bool__(self):
        return self.status < OperationStatus.INTERRUPTED


class Operation(DebuggableMixin):
    """A Mitten Operation requires:
        - A scratch space to work in, along with a namespace to use under the scratch space
        - A set of Operations, representing immediate dependencies of this Operation that are required for this
          Operation to run successfully. This can be empty if none exist.

    An Operation, when run, must return a dictionary or None.

    An Operation is treated as successfully run if no exceptions are raised during Operation.run().

    Children of a parent Operation must include *all* of the parent's `output_keys`. If an Operation has multiple parent
    Operations that has a key with the same name (i.e. 'foo') but should have different values, the child Operation
    should output a dict, where key 'foo' has a value that is a dictionary, where the keys are the parent classes
    themselves. For example:
        {
            'foo': {
                ParentOperation1: <some value>,
                ParentOperation2: <some other value>,
            }
        }
    """

    @classmethod
    def output_keys(cls):
        """The list of output keys that are mandatory for this Operation to return in `run()`. If no keys are necessary,
        this method should return an empty list.
        """
        return []

    @abstractclassmethod
    def immediate_dependencies(cls):
        """Get the immediate dependencies of this operation.

        Returns:
            A set of classes. Each class in this set must be a subclass of Operation
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, scratch_space, dependency_outputs):
        """Runs the Operation.

        Args:
            scratch_space (ScratchSpace): The ScratchSpace for this Operation.
            dependency_outputs (Dict[str, Any]): A dict of named objects from outputs of this Operation's
                                                 dependencies.

        Returns:
            If this Operation has objects that are consumed by upstream Operations, output a dict of str to object,
            where the key is a unique name for this object. Otherwise, a bool indicating this Operation's success.
        """
        raise NotImplementedError

    @classmethod
    def _verify(cls):
        # Check if output_keys is a superset of all parent output_keys.
        parents = inspect.getmro(cls)[1:]  # First elem is always `cls`.
        oks = set(cls.output_keys())

        for parent_cls in parents:
            if parent_cls is Operation:
                continue

            if not issubclass(parent_cls, Operation):
                continue

            parent_oks = set(parent_cls.output_keys())
            if not oks.issuperset(parent_oks):
                diff = parent_oks - oks
                raise MissingParentOutputKey(cls, parent_cls, diff)

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls._verify()

    @classmethod
    def implements_op(cls, other: type = None) -> bool:
        """Checks if `cls` implements `other` as a subclass. If `other` is not provided, then checks if `cls` extends at
        least 1 Operation implementation.

        If `other` is provided but is not an Operation, this method will by default return `False`, even if it is a
        superclass of `cls`. Callers of this method should handle this case separately.

        Args:
            other (type): The class to check is a parent class of `cls`. If not provided, checks if `cls` extends at
                          least 1 Operation implementation.

        Returns:
            bool: True if the above condition is True, otherwise False.
        """
        parents = inspect.getmro(cls)[1:]  # First elem is always `cls`.
        if other and not issubclass(other, Operation):
            return False

        for p in parents:
            if p is Operation or not issubclass(p, Operation):
                continue

            if other and p is other:
                return True

            if not other and issubclass(p, Operation):
                return True
        return False

    @classmethod
    def implements(cls) -> List[type]:
        """Similar to `implements_op`, but instead of returns a list of all Operations this `cls` implements.

        Returns:
            List[type]: List of all Operations that are superclasses of `cls`.
        """
        parents = inspect.getmro(cls)[1:]  # First elem is always `cls`.

        impls = []
        for p in parents:
            if p is Operation or not issubclass(p, Operation):
                continue

            if issubclass(p, Operation):
                impls.append(p)
        return impls
