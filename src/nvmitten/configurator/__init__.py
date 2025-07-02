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
import argparse
import ast
from collections import defaultdict
import dataclasses
import inspect
import functools
from typing import Any, Callable, Dict, List, Set, Tuple, Type, Union

from .fields import Field
from .configurable import Configuration, ConfigurationIndex, _autoconf_state
from ..pipeline import Operation


class _HelpInfo:
    def __init__(self):
        self.conf_deps = defaultdict(set)

    def add_configurator_dependency(self, user_cls: Type, dep_cls: Type):
        """Add a configurator dependency between two classes.

        Args:
            user_cls (Type): The class that depends on another configurable class
            dep_cls (Type): The class that is depended on by `user_cls`
        """
        self.conf_deps[user_cls].add(dep_cls)

    def has(self, cls: Type) -> bool:
        return cls in self.conf_deps

    def add_configurator_dependencies(self, user_cls: Type, deps: Set[Type]):
        """Add configurator dependencies between a user class and a set of classes.

        Args:
            user_cls (Type): The class that depends on other configurable classes
            deps (Set[Type]): The set of classes that `user_cls` depends on
        """
        self.conf_deps[user_cls].update(deps)

    def add_configurable_operation(self, op: Type[Operation]):
        """Automatically add the dependencies of a configurable operation to the help info.

        Args:
            op (Operation): The operation to add
        """
        if not issubclass(op, Operation):
            raise ValueError(f"{op} is not a subclass of Operation")

        if op.immediate_dependencies():
            self.add_configurator_dependencies(op, set(op.immediate_dependencies()))

    def get_all_dependencies(self, cls: Type) -> Set[Type]:
        """Get all configurator dependencies for a given class.

        Args:
            cls (Type): The class to get dependencies for

        Returns:
            Set[Type]: A set of all configurator dependencies for the given class
        """
        seen = set()
        bag = [cls]
        while bag:
            curr = bag.pop(0)
            if curr not in seen:
                seen.add(curr)
                if curr in self.conf_deps:
                    bag.extend(list(self.conf_deps[curr]))
        return seen

    def build_help_string(self, classes: Union[Type, List[Type]]) -> str:
        """Build a help string for a given class.

        Args:
            classes (Type): The class or classes to build a help string for

        Returns:
            str: A help string for the given class(es)
        """
        used_fields = set()

        if not isinstance(classes, list):
            classes = [classes]

        for cls in classes:
            deps = self.get_all_dependencies(cls)
            for configurable in deps:
                if "_mitten_bound_fields" not in vars(configurable):
                    continue

                for f in configurable._mitten_bound_fields.keys():
                    used_fields.add(f)

        fgroups = defaultdict(list)
        for f in used_fields:
            fgroups[f._created_in].append(f)

        parser = argparse.ArgumentParser(allow_abbrev=False,
                                         argument_default=argparse.SUPPRESS,
                                         add_help=False)
        parser_groups = {}
        for group, fields in fgroups.items():
            if group not in parser_groups:
                # Find docstring
                with group.open(mode='r') as fobj:
                    tree = ast.parse(fobj.read())
                    gdesc = ast.get_docstring(tree)

                parser_groups[group] = parser.add_argument_group(str(group), gdesc)

            arg_group = parser_groups[group]
            for field in fields:
                field.add_to_argparse(arg_group)
        return parser.format_help()


# Singleton instance of _HelpInfo
HelpInfo = _HelpInfo()


def _parse_init_kwarg(cls: Type, s: str) -> Tuple[Type, Any]:
    """Parse a keyword argument from a class's __init__ method.

    Args:
        cls (Type): The class to parse the keyword argument from
        s (str): The name of the keyword argument to parse

    Returns:
        Tuple[Type, Any]: A tuple containing the type annotation and default value of the keyword argument

    Raises:
        KeyError: If the keyword argument is not found or is a non-init dataclass field
    """
    if dataclasses.is_dataclass(cls):
        for f in dataclasses.fields(cls):
            if f.name == s:
                if not f.init:
                    raise KeyError(f"'{s}' is a non-init dataclass field to {cls}")
                return (f.type, f.default)
        raise KeyError(f"'{s}' is not a member of dataclass {cls}")
    else:
        if "_init_signature" in vars(cls):
            sig = cls._init_signature
        else:
            sig = inspect.signature(cls.__init__)

        if s not in sig.parameters:
            raise KeyError(f"'{s}' is not a value parameter to {cls}.__init__({sig})")

        param = sig.parameters[s]
        return (param.annotation, param.default)


def _build_call_args(cls, config: Configuration = None, **kwargs):
    """Build the arguments to be passed to a class's __init__ method.

    Args:
        cls: The class to build arguments for
        config (Configuration, optional): Configuration to use for building arguments
        **kwargs: Additional keyword arguments to include

    Returns:
        Dict[str, Any]: Dictionary of arguments to be passed to __init__
    """
    if "_mitten_bound_fields" not in vars(cls):
        return kwargs

    if config is None:
        config = Configuration()

    call_args = config.configure(cls)
    call_args.update(kwargs)
    return call_args


def from_fields(cls, *args, config: Configuration = None, **kwargs):
    """Create an instance of a class using field-based configuration.

    Args:
        cls: The class to instantiate
        *args: Positional arguments to pass to the constructor
        config (Configuration, optional): Configuration to use for field values
        **kwargs: Additional keyword arguments to pass to the constructor

    Returns:
        An instance of the class configured with the provided fields and arguments
    """
    return cls(*args, **_build_call_args(cls, config=config, **kwargs))


def configurable_setup(cls):
    """Sets up the required attributes for Mitten configurator to work.

    Args:
        cls (Type): The class to set up
    """
    def set_if_not_exists(attr, val, ignore_parents: bool = False):
        if ignore_parents:
            cond = attr not in vars(cls)
        else:
            cond = not hasattr(cls, attr)

        if cond:
            setattr(cls, attr, val)

    set_if_not_exists("_original_init", cls.__init__, ignore_parents=True)
    set_if_not_exists("_init_signature", inspect.signature(cls.__init__), ignore_parents=True)
    set_if_not_exists("_mitten_bound_fields", dict(), ignore_parents=True)
    set_if_not_exists("from_fields", classmethod(from_fields))


def bind(field: Field,
         arg_name: str = None,
         explicit: bool = True):
    """Decorator to bind a field to a kwarg of a class's __init__ method. If the name of the parameter is not provided,
    this method will assume the name of the parameter is the same as the name of the field.

    Args:
        field (Field): The Field to bind to a keyword argument
        arg_name (str): The name of the keyword argument to bind to. If not provided, uses the name of the Field.
        explicit (bool): Whether or not the field is explicitly declared in the signature of __init__. If True, will
                         parse __init__ to verify this is the keyword argument exists. (Default: True)
    """
    def _f(cls, _field: Field = None, kw: str = None):
        """Internal function to perform the actual binding of a field to a class.

        Args:
            cls: The class to bind the field to
            _field (Field, optional): The field to bind
            kw (str, optional): The keyword argument name to bind to

        Returns:
            The modified class with the field binding

        Raises:
            KeyError: If the parameter is already bound to another field
        """
        configurable_setup(cls)

        if kw is None:
            kw = _field.name

        if explicit:
            _parse_init_kwarg(cls, kw)  # Implicitly throws assertion error

        if _field in cls._mitten_bound_fields:
            _v = cls._mitten_bound_fields[_field]
            raise KeyError(f"Param {kw} of {cls} is already bound to {_v}")

        cls._mitten_bound_fields[_field] = kw

        # If the class is a subclass of Operation, automatically add it to HelpInfo
        if issubclass(cls, Operation) and not HelpInfo.has(cls):
            HelpInfo.add_configurable_operation(cls)
        return cls
    return functools.partial(_f, _field=field, kw=arg_name)


def autoconfigure(cls):
    """Decorator to automatically configure a class using the current configuration context.

    This decorator creates a subclass that automatically applies configuration from the current
    context when instantiating the class. It also handles inheritance of field bindings for
    dataclass subclasses.

    Args:
        cls: The class to be configured automatically

    Returns:
        A subclass of the input class with automatic configuration support
    """
    configurable_setup(cls)

    @functools.wraps(cls.__init__)
    def auto_init(self, *args, **kwargs):
        """Initialize an instance with automatic configuration.

        If a configuration context is active, it will be used to configure the instance.
        Otherwise, the instance will be initialized with the provided arguments.

        Args:
            *args: Positional arguments to pass to the parent class's __init__
            **kwargs: Keyword arguments to pass to the parent class's __init__
        """
        if (conf := _autoconf_state.get()) is not None:
            kwargs = _build_call_args(cls, config=conf, **kwargs)

        cls._original_init(self, *args, **kwargs)
    cls.__init__ = auto_init
    return cls
