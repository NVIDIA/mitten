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


from types import ModuleType
from typing import List
import importlib
import sys


class ScopedImporter:
    """Scope where sys.path is temporarily modified. Any imports inside the scope will still be visible globally, but
    are imported from the modified sys.path.

    After the scope ends, sys.path will be reverted back to its original value before the scope was created. This is NOT
    thread-safe.
    """

    def __init__(self, import_path: List[str]):
        """Creates a ScopedImporter where sys.path equals `import_path`.
        """
        self.old_sys_path = None
        self.import_path = import_path

    def __enter__(self):
        self.old_sys_path = sys.path[:]
        sys.path = self.import_path[:]
        return self

    def __exit__(self, type, value, traceback):
        sys.path = self.old_sys_path[:]

    def path_as_string(self):
        return ":".join(x for x in self.import_path if len(x) > 0)


def import_from(import_path: List[str], module_name: str) -> ModuleType:
    """Imports a module from a specified import path. The import path must be a valid value for `sys.path` (i.e. a list
    of strings).

    Args:
        import_path (List[str]): The value to use as sys.path
        module_name (str): The module to import

    Returns:
        ModuleType: The imported module
    """
    with ScopedImporter(import_path):
        mod = importlib.import_module(module_name)
    return mod
