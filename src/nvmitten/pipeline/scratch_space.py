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
from os import PathLike
from pathlib import Path


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
