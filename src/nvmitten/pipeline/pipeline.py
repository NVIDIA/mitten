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
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import graphlib
import logging
import shutil

from .errors import *
from ..memory import Memory


class ScratchSpace:
    """Marks a directory to be used as the scratch space for a pipeline. Note that this class intentionally does not
    include a method to delete the contents of a ScratchSpace, as it is common for the ScratchSpace to contain important
    logs, large downloaded files such as datasets, or important DL model checkpoints. As such, a clear() or delete()
    method is not provided to help prevent mishaps or accidents.

    If you know what you're doing and would like to clear the contents of a scratch space, a deletion method is easily
    implemented with shutil.rmtree.
    """

    def __init__(self, path: PathLike):
        """Creates a ScratchSpace that corresponds to the given path.

        Args:
            path (PathLike): The path to the directory to use as the scratch space.
        """
        self.path = path
        if not isinstance(path, Path):
            self.path = Path(path)

    def create(self):
        if not self.path.exists():
            self.path.mkdir(parents=True)

    def working_dir(self, namespace: str = "") -> Path:
        """Gets a working dir from the ScratchSpace under a namespace. In terms of the directory structure, the
        namespace is simply a subdirectory. Also creates this directory if it does not yet exist.

        Args:
            namespace (str): The namespace of the working dir under this ScratchSpace. (Default: "")

        Returns:
            Path: The path to the working directory.
        """
        wd = self.path / Path(namespace)
        if not wd.exists():
            wd.mkdir(parents=True)
        return wd


@dataclass(frozen=True)
class BenchmarkMetric:
    """Defines a metric unit used for a Benchmark. Two metrics can only be compared if their units are equal.
    """

    unit: str
    bigger_is_better: bool = True

    def __eq__(self, other: Any) -> bool:
        if self.__class__ is other.__class__:
            return self.unit.upper() == other.unit.upper() and \
                    self.bigger_is_better == other.bigger_is_better
        else:
            return NotImplemented


class Impl:
    """Represents an Operation with given outputs. Can be used by Operations to indicate requiring a dependency that
    produces a certain output, but is agnostic to the implementation of that dependency.
    """

    @classmethod
    def outputs(cls):
        """Returns an Iterable of strings, representing the keys that are required to be in the output dictionary of any
        Operation's run() method that is an implementation of this Impl.

        If no outputs are required, this method should return None or an empty Iterable.
        """
        raise NotImplementedError


class Operation(ABC):
    """A Mitten Operation requires:
        - A scratch space to work in, along with a namespace to use under the scratch space
        - A set of Operations, representing immediate dependencies of this Operation
    """

    @classmethod
    def implements(cls):
        """Gets the Impl, if any, that this Operation implements.
        """
        return None

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


class Pipeline:
    """A Mitten pipeline consists of the following:
        1. A graph of operations, each representing one step or phase of the pipeline.
        2. A pipeline-unified scratch space for the operations of the pipeline to share artifacts with
    """

    def __init__(self, scratch_space, operations, config):
        self.scratch_space = scratch_space
        self.operations = operations
        self.config = config

        self.output_node = None

    def topo_sort(self) -> Tuple[Tuple[Operation, ...], Dict[Impl, Operation]]:
        """Topologically sorts the operations and returns the sorted order of the operations, as well as a mapping of
        found Impls to their implemented Operations.

        Returns:
            Tuple[Operation, ...]: A tuple representing the ordering of operations sorted topologically by dependencies.
            Dict[Impl, Operation]: A map implementing Impls to the Operation implementing them.
        """
        # Create a mapping for implementations
        implementations = dict()
        for op in self.operations:
            if impl := op.implements():
                if not issubclass(impl, Impl):
                    raise InvalidImplError(op, impl)
                if impl in implementations:
                    raise TooManyImplementationsError(impl, [implementations[impl], op])
                implementations[impl] = op

        g = dict()
        for op in self.operations:
            _deps = op.immediate_dependencies()
            if _deps is None:
                g[op] = set()
                continue

            deps = set()
            for dep in _deps:
                if issubclass(dep, Impl):
                    if dep not in implementations:
                        raise ImplementationNotFoundError(op, impl)
                    dep = implementations[dep]
                deps.add(dep)
            g[op] = set(deps)
        ts = graphlib.TopologicalSorter(g)
        return tuple(ts.static_order()), implementations

    def run(self):
        """Runs the pipeline.

        Stores an internal cache of all Operation outputs which is used to forward inputs to dependent Operations.

        Returns:
            obj: The return value of this pipeline's output. If no output Operation is marked, the final executed
            Operation is treated as the output Operation.
        """
        if self.scratch_space:
            self.scratch_space.create()

        cache = dict()
        ordered, implementations = self.topo_sort()
        run_output = None
        for op in ordered:
            op_inputs = dict()
            if op.immediate_dependencies() is not None:
                for dep in op.immediate_dependencies():
                    if issubclass(dep, Impl):
                        # If dependency is an Impl, forward the Impl's expected outputs.
                        _in = dict()
                        for k in dep.outputs():
                            _in[k] = cache[implementations[dep]][k]  # If k isn't in cache this will error.
                        op_inputs[dep] = _in
                    else:
                        if dep in cache and cache[dep]:
                            op_inputs[dep] = cache[dep]
            instance = op(**self.config.get(op, dict()))
            run_output = instance.run(self.scratch_space, op_inputs)
            if not run_output:
                raise FailedOperationError(op, run_output)
            if isinstance(run_output, dict):
                cache[op] = run_output

            # Early stop if we hit output node in our topo-sorted order.
            if op is self.output_node:
                return run_output

        if self.output_node is None:
            return run_output  # Output the final output of the graph
        else:
            return cache[self.output_node]

    def mark_output(self, op: Operation):
        """Marks an Operation for output.

        Important notes:
         - Mitten pipelines do not support marking multiple operations as outputs.
         - A pipeline's output should not mention the Operations done during the pipeline.
         - The output is formatted the same way as an Operation's output: a flat
           dictionary mapping named keys to values.
         - It is also not always the case that all the outputs of an output node are used as output for the pipeline.

        To use outputs from multiple Operations as the pipeline output, or to filter out unwanted outputs, create an
        "Aggregation Operation" which depends on the necessary operations and outputs the required outputs for the
        pipeline.

        See `examples/1_PyTorch` for an example of this.

        Args:
            op (Operation): The Operation (class value) to mark as output.

        Raises:
            ValueError: If `op` is not in `self.operations`.
        """
        if op not in self.operations:
            raise KeyError(f"Cannot mark not-in-pipeline {op} as output.")
        self.output_node = op
