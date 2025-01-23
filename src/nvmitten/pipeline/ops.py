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


__doc__ = """Implementations of commonly-used or generic operations."""


from __future__ import annotations
from os import PathLike
from pathlib import Path

import logging
import shutil

from .pipeline import Operation
from ..utils import run_command


class ExternalCommandOp(Operation):
    """Base class for an operation that executes a command and returns. Subclasses must still implement the
    `.implements()` and `.immediate_dependencies()` classmethods.
    """

    def __init__(self, cmd: str = None, tee: bool = False, verbose: bool = False):
        """Constructs an ExternalCommandOp.

        Args:
            cmd (str): The command to execute
        """
        # Verify that the command exists
        if not shutil.which(cmd):
            raise FileNotFoundError(f"Command '{cmd}' does not exist")
        self.cmd = cmd
        self.tee = tee
        self.verbose = verbose

    def run(self, scratch_space, dependency_outputs):
        output = run_command(self.cmd,
                             get_output=True,
                             tee=self.tee,
                             verbose=self.verbose)
        return {"output": output}
