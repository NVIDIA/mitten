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
from collections import UserDict
from contextlib import contextmanager
from contextvars import ContextVar
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, List, Set

import dataclasses
import importlib
import logging
import sys

from .fields import Field, parse_fields
from ..importer import import_from
from ..tree import Tree, Traversal


_autoconf_state = ContextVar("autoconf_state", default=None)


class Configuration(UserDict):
    """A dictionary-like class for managing configuration settings.

    This class extends UserDict to provide configuration management functionality,
    including subset extraction and configuration application to classes.
    """

    def subset(self, ctx: List[str], applicable: List[Field]) -> Configuration:
        """Extracts a sub-dictionary of only the provided applicable fields from the context specified.

        Args:
            ctx (List[str]): The context path to extract configuration from (used as a prefix for the applicable keys)
            applicable (List[Field]): List of fields to include in the subset

        Returns:
            Configuration: A new Configuration instance containing only the specified fields

        Raises:
            KeyError: If the specified context path is invalid
        """
        # Get to context
        d = self.data
        for k in ctx:
            if k not in d:
                raise KeyError(f"Invalid context {ctx} for Configuration {self._id}")
            d = d[k]

        # Get subset
        return Configuration({f.name: d[f.name] for f in applicable if f.name in d})

    def configure(self, cls) -> Dict[str, Any]:
        """Configures a class using the stored configuration values.

        Args:
            cls: The class to configure

        Returns:
            Dict[str, Any]: Dictionary of configuration values to be applied to the class

        Raises:
            ValueError: If a required field (disallow_default=True) is not specified
        """
        if "_mitten_bound_fields" not in vars(cls):
            logging.warn(f"Class {cls} is being configured, but has no bound fields")
            return dict()

        to_parse = cls._mitten_bound_fields

        # Support for inherited dataclasses.
        # Dataclass subclasses do not auto-invoke super.__init__ and instead generate a "bigger" __init__ for the
        # subclass with all the parent's dataclass fields.
        # In this case we have to traverse the MRO to search for configurable parents and add their fields.
        if dataclasses.is_dataclass(cls):
            for k in cls.mro()[1:]:
                if not dataclasses.is_dataclass(k):
                    continue

                for parent_field, parent_kw in vars(k).get("_mitten_bound_fields", dict()).items():
                    if parent_field not in to_parse:
                        to_parse[parent_field] = parent_kw

        cli_args = parse_fields(to_parse)
        kwargs = dict()
        for f, kw in to_parse.items():
            if f.disallow_default and f not in self.data and f.name not in cli_args:
                # Check for disallow_default
                raise ValueError(f"{cls}::{f.name} disallows default values, but is not specified in configuration or CLI")
            elif f.name in cli_args:
                # Apply CLI override
                kwargs[kw] = cli_args[f.name]
            elif f in self.data:
                # Apply this configuration
                kwargs[kw] = self.data[f]
        return kwargs

    @contextmanager
    def autoapply(self):
        """Context manager for automatically applying configuration.

        This context manager sets the current configuration as the active configuration
        for the duration of the context block.

        Yields:
            None: No value is yielded
        """
        token = _autoconf_state.set(self)
        try:
            yield None  # No value to propagate
        finally:
            _autoconf_state.reset(token)


class ConfigurationIndex:
    """A class for managing and indexing multiple configurations.

    This class provides functionality to register, retrieve, and load configurations
    from various sources including TOML files and Python modules.
    """

    def __init__(self, index_name: str = "ConfigurationIndex"):
        """Initialize a new ConfigurationIndex.

        Args:
            index_name (str, optional): Name of the index. Defaults to "ConfigurationIndex".
        """
        self._data = Tree(index_name, None)

    def register(self, keyspace, config):
        """Register a configuration in the index.

        Args:
            keyspace: The key path where the configuration should be stored
            config: The configuration to register
        """
        self._data[keyspace] = config

    def get(self, keyspace: List[str], prefix: List[str] = None):
        """Retrieve a configuration from the index.

        Args:
            keyspace (List[str]): The key path to retrieve
            prefix (List[str], optional): Optional prefix to prepend to the keyspace

        Returns:
            The configuration if found, None otherwise
        """
        if prefix:
            keyspace = prefix + keyspace
        if keyspace in self._data:
            return self._data[keyspace].value
        return None

    def load_module(self,
                    module_path: str,
                    prefix: List[str] = None,
                    process_key: Callable[[List[str]], None] = None):
        """Loads configurations from a Python module into the index.

        Args:
            module_path (str): Path to the Python module
            prefix (List[str], optional): Initial keyspace to traverse from when adding configurations
            process_key (Callable[[List[str]], None], optional): Function to process keys before insertion

        Raises:
            ValueError: If the module has no exported configurations
        """
        d = self._data
        if prefix:
            if prefix not in d:
                d = d.insert_value(prefix[-1],
                                   None,
                                   keyspace=prefix[:-1],
                                   create_parents=True)
            else:
                d = d[prefix]

        mod = importlib.import_module(module_path)
        if not hasattr(mod, "EXPORTS"):
            raise ValueError(f"Module {module_path} has no exported configurations")

        for k, dat in mod.EXPORTS.items():
            if process_key:
                key = process_key(k)
            else:
                key = k

            conf = Configuration()
            for f, val in dat.items():
                if f.from_string and isinstance(val, str):
                    val = f.from_string(val)
                conf[f] = val
            d[key] = conf
