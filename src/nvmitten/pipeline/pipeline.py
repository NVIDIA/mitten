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
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import graphlib
import logging
import time
import traceback

from .errors import *
from .operation import *
from .scratch_space import ScratchSpace
from ..debug import DebugManager, DebuggableMixin


class Pipeline(DebuggableMixin):
    """A Mitten pipeline consists of the following:
        1. A graph of operations, each representing one step or phase of the pipeline.
        2. A pipeline-unified scratch space for the operations of the pipeline to share artifacts with
    """

    def __init__(self, scratch_space, operations, config):
        self.scratch_space = scratch_space
        self.operations = operations
        self.config = config

        self.output_node = None
        self._cache = None

    def topo_sort(self) -> Tuple[Tuple[Operation, ...], Dict[Operation, Operation]]:
        """Topologically sorts the operations and returns the sorted order of the operations, as well as a mapping of
        found implementations of dependency Operations.

        Returns:
            Tuple[Operation, ...]: A tuple representing the ordering of operations sorted topologically by dependencies.
            Dict[Operation, Operation]: A map of Operation-to-be-implemented to the implementation.
        """
        implementations = dict()
        for op in self.operations:
            for i in op.implements():
                # TODO: Implement a way to configure this. For now just print a warning.
                if i in implementations:
                    logging.warning(f"Multiple implementations of {i} found. Defaulting to {implementations[i]}.")
                else:
                    implementations[i] = op

        g = dict()
        for op in self.operations:
            requires = op.immediate_dependencies()
            if requires is None:
                g[op] = set()
                continue

            deps = set()
            for dep in requires:
                if dep in self.operations:
                    # dep is directly satisfied, use directly
                    deps.add(dep)

                    # Warn if a child implementation was given and unused.
                    if dep in implementations:
                        impl = implementations[dep]
                        msg = f"Operation {dep} has implementation {impl}, but is directly satisfied. " \
                              f"Using {dep} instead."
                        logging.warn(msg)
                        implementations.pop(dep)
                else:
                    # dep is not directly satisfied, search for implementation
                    if dep not in implementations:
                        raise ImplementationNotFoundError(op, dep)
                    deps.add(implementations[dep])
            g[op] = set(deps)

        ts = graphlib.TopologicalSorter(g)
        return tuple(ts.static_order()), implementations

    def run(self, early_stop: bool = False):
        """Runs the pipeline.

        Stores an internal cache of all Operation outputs which is used to forward inputs to dependent Operations.

        Args:
            early_stop (bool): If True, will stop pipeline execution as soon as the output node is reached and will
                               return immediately. This is not recommended if all Operations are expected to run at
                               least once, as the topologicaly sorted order of Operations may cause some Operations to
                               not run. (Default: False)

        Returns:
            obj: The return value of this pipeline's output. If no output Operation is marked, the final executed
            Operation is treated as the output Operation.
        """
        if self.scratch_space:
            self.scratch_space.create()

        self._cache = dict()
        ordered, implementations = self.topo_sort()

        run_output = None
        output_reached = False
        for op in ordered:
            # Build inputs based on cache
            op_inputs = dict()
            if op.immediate_dependencies() is not None:
                for dep in op.immediate_dependencies():
                    if dep in implementations:
                        impl = implementations[dep]

                        # Only take the set of keys that the specific dependency requires
                        _in = dict()
                        for k in dep.output_keys():
                            _in[k] = self._cache[impl].value[k]  # If k isn't in cache this will implicitly error.
                        op_inputs[dep] = _in
                    else:
                        if dep in self._cache:
                            op_inputs[dep] = self._cache[dep].value

            status = None
            run_output = None
            exc = None
            time_start = time.time_ns()
            try:
                if not output_reached:
                    instance = op(**self.config.get(op, dict()))
                    run_output = instance.run(self.scratch_space, op_inputs)
                    exc = None
                    status = OperationStatus.PASSED
                else:
                    status = OperationStatus.SKIPPED
            except KeyboardInterrupt as _exc:
                exc = KeyboardInterrupt
                status = OperationStatus.INTERRUPTED
            except Exception as _exc:
                exc = _exc
                status = OperationStatus.FAILED
            finally:
                tb = None
                if exc is not None:
                    tb = traceback.TracebackException.from_exception(exc)

                res = OperationResult(status,
                                      run_output,
                                      time_start,
                                      exception=exc,
                                      trace=tb)
                self._cache[op] = res

                # Re-surface exception.
                # TODO: Let DebugHandler handle retrieving the OperationResult cache and dumping it.
                if exc is not None:
                    raise exc

            # Early stop if we hit output node in our topo-sorted order.
            if early_stop and op is self.output_node:
                output_reached = True

        if self.output_node is None:
            return run_output  # Output the final output of the graph
        else:
            return self._cache[self.output_node].value

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
