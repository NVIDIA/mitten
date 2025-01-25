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

import logging
import os
import platform
import re
import subprocess
import sys

from git import RemoteProgress
from glob import glob
from typing import Any, Dict, Final, List, Optional, Set
from tqdm import tqdm


UNSET_VALUE: Final[Any] = object()
"""object: Used as a magic value denoting that a field is unset, if None has a special meaning and cannot be used."""


def run_command(cmd: str,
                get_output: bool = False,
                tee: bool = True,
                custom_env: Optional[Dict[str, str]] = None,
                verbose: bool = True) -> List[str]:
    """
    Runs a command and returns its output.

    Args:
        cmd (str): The command to run.
        get_output (bool): If False, run the command without capturing output. (Default: False)
        tee (bool): If True, prints output to stdout during execution in addition to returning it. This is essentially
                    the `tee` utility in Bash. (Default: True)
        custom_env (Dict[str, str]): If set, used as the custom environment for the command being run. (Default: None)
        verbose (bool): If True, log the command being run and any custom environment used. (Default: True)

    Returns:
        List[str]: A list of the output lines of the command run

    Raises:
        subprocess.CalledProcessError: If the command returns a non-zero exit code
    """
    if verbose:
        logging.info("Running command: {:}".format(cmd))

    output = []
    if custom_env is not None:
        if verbose:
            logging.info(f"Overriding Environment: {custom_env}")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, env=custom_env)
    else:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

    for line in iter(p.stdout.readline, b""):
        line = line.decode("utf-8")
        if tee:
            sys.stdout.write(line)
            sys.stdout.flush()
        if get_output:
            output.append(line.rstrip("\n"))
    ret = p.wait()
    if ret == 0:
        return output
    else:
        raise subprocess.CalledProcessError(ret, cmd)


def dict_get(d, key, default=None):
    """Return non-None value for key from dict. Use default if necessary."""
    val = d.get(key, default)
    return default if val is None else val


def dict_eq(d1: Dict[str, Any], d2: Dict[str, Any], ignore_keys: Optional[Set[str]] = None) -> bool:
    """Compares 2 dictionaries, returning whether or not they are equal. This function also supports ignoring keys for
    the equality check. For example, if d1 is {'a': 1, 'b': 2} and d2 is {'a': 1, 'b': 3, 'c': 1}, if ignore_keys is set
    to {'b', 'c'}, this method will return True.
    While this method supports dicts with any type of keys, it is recommended to use strings as keys.

    Args:
        d1 (Dict[str, Any]): The first dict to be compared
        d2 (Dict[str, Any]): The second dict to be compared
        ignore_keys (Set[str]): If set, will ignore keys in this set when doing the equality check

    Returns:
        bool: Whether or not d1 and d2 are equal, ignore the keys in `ignore_keys`
    """
    def filter_dict(d): return {k: v for k, v in d.items() if k not in ignore_keys}
    return filter_dict(d1) == filter_dict(d2)


class GitPyTqdm(RemoteProgress):
    """Utility wrapper around tqdm for use with GitPython operations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pbar = tqdm(*args, **kwargs)

    def update(self, op_code, cur_count, max_count=None, message=''):
        self.pbar.total = max_count
        self.pbar.n = cur_count
        self.pbar.refresh()
