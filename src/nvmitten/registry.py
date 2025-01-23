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
from typing import Any, Callable, Iterable, Optional, Union

import functools

from .utils import UNSET_VALUE
from .tree import Tree


class RegistryTree(Tree):
    """Tree implementation where trees are mainly referenced from the root using __getattr__ and __setattr__
    with a keyspace.

    insert_value behavior now creates missing nodes in a walk if provided (similar to `mkdir -p` behavior.
    """

    @staticmethod
    def _default_value_factory():
        """The "default" default value implementation
        """
        return UNSET_VALUE

    def __init__(self,
                 name: str,
                 value: Any,
                 children: Optional[Iterable[Tree]] = None,
                 default_value_factory: Callable[[], Any] = None):
        """Creates a Registry node.

        Args:
            name (str): ID used to query for this node. Names must be unique among other immediate child nodes of the
                        same parent node.
            value (Any): The value to store for the node.
            children (Iterable[Tree]): If set, initializes this nodes children to this. Otherwise, initialized with no
                                       child nodes. (Default: None)
            default_value_factory (Callable[[], Any]): Function that returns a default value for a RegistryTree node.
                                                       An important note is that the default value must not change
                                                       between function calls, and that the default value is assumed to
                                                       be unused in normal usage, and indicates an unset value.

                                                       For example, in a RegistryTree of filepaths, the empty string or
                                                       None are suitable default values.

                                                       If None, defaults to RegistryTree._default_value_factory.
                                                       (Default: None)
        """
        super().__init__(name, value, children=children)
        if default_value_factory is None:
            self.default_value_factory = RegistryTree._default_value_factory
        else:
            self.default_value_factory = default_value_factory

    def default_factory(self, key: str) -> RegistryTree:
        """Wrapper around default_value_factory to use as the default_factory for _KeyBasedDefaultDict.

        Creates a Tree node with the same class as self with `key` as `name`, the return value of
        `self.default_value_factory` as `value`, and the same `default_value_factory` as `self`, propagating itself.

        Args:
            key (str): The key with missing value

        Returns:
            RegistryTree: The new DefaultTree Node to insert.
        """
        return self.__class__(key, self.default_value_factory(), default_value_factory=self.default_value_factory)

    def insert_value(self, name: str, value: Any, keyspace: Optional[Iterable[str]] = None) -> Tree:
        """Inserts a value into this node. The created child node will be the same class as its immediate parent.

        Args:
            name (str): The name of the node to be created
            value (Any): The value to insert
            keyspace (Iterable[str]): If specified, instead inserts the value in the node at the end of the walk
                                      starting from `self` represented by the names of the nodes given in
                                      `keyspace`. (Default: None)

        Returns:
            Tree: The Tree that was created

        Raises:
            KeyError: If the walk specified by `keyspace` is an invalid walk of the Tree
        """
        curr = self
        if keyspace:
            for key in keyspace:
                if key in curr.children:
                    curr = curr.children[key]
                else:
                    new_node = self.default_factory(key)
                    curr.add_child(new_node)
                    curr = new_node
        new_node = curr.__class__(name, value, default_value_factory=self.default_value_factory)
        curr.add_child(new_node)
        return new_node


class Registry:
    """Used as a bookkeeping data structure to store instances or subclasses of a particular class.

    Registry provides a member variable `self.register` that serves as a shorthand to call either `register_instance` or
    `register_class`.

    If a Registry instance has `register_instances=True`, then `register` simply points at `register_instance`. However,
    if a Registry instance has `register_instances=False`, then `register` is a decorator:
        @register(keyspace) -> func(to_register) -> to_register
    """

    def __init__(self,
                 registerable_class: type,
                 register_instances: bool = True):
        """Creates a new Registry for a given type.

        Args:
            registerable_class (type): The class type for this Registry. All registered objects must be instances of
                                       this type, or if `register_instances` is False, must be subclasses.
                                       If `register_instances` is True, `registerable_class` must implement `__hash__`.
            register_instances (bool): Whether or not registered objects are types or instances. (Default: True)
        """
        self.registry = RegistryTree(registerable_class.__name__, registerable_class)
        self.registerable_class = registerable_class
        self.register_instances = register_instances
        if register_instances:
            self.register = self.register_instance
        else:
            def _decorator(keyspace):
                def _f(o):
                    return self.register_class(keyspace, o)
                return _f
            self.register = _decorator

    def register_instance(self, keyspace: Union[str, Iterable[str]], o: Any) -> Any:
        """Registers a new instance of `registerable_class` under this registry under a given keyspace.

        Args:
            keyspace (Iterable[str]): The keyspace (or key if just a str is provided) to register this instance as
            o (Any): The object instance to register

        Returns:
            Any: Returns the input `o`
        """
        if not isinstance(o, self.registerable_class):
            raise ValueError(f"Cannot register object of type {o.__class__.__name__} under registry of "
                             f"{self.registerable_class.__name__}.")
        self.registry[keyspace] = o
        return o

    def register_class(self, keyspace: Union[str, Iterable[str]], o: type) -> Any:
        """Registers a subclass of `registerable_class` under this registry under a given keyspace.

        Args:
            keyspace (Iterable[str]): The keyspace (or key if just a str is provided) to register this class as
            o (type): The class to register

        Returns:
            Any: Returns the input `o`
        """
        if not issubclass(o, self.registerable_class):
            raise ValueError(f"Cannot register class {o.__name__} under registry of "
                             f"{self.registerable_class.__name__}.")
        self.registry[keyspace] = o
        return o

    def get(self, keyspace: Union[str, Iterable[str]]) -> Any:
        """Retrieves data from the registry given the data's keyspace.

        Args:
            keyspace (Iterable[str]): The keyspace (or key if just a str is provided) to retrieve from

        Returns:
            Any: The data value at the keyspace

        Raises:
            KeyError: If the keyspace is invalid
        """
        if isinstance(keyspace, str):
            keyspace = [keyspace]

        node = self.registry.get(keyspace)
        if len(node.children) == 0:
            if node.value is UNSET_VALUE:  # This is known because of the default RegistryTree constructor
                raise KeyError(keyspace)
            return node.value
        else:
            return node.get_children()

    def unregister(self, keyspace: Union[str, Iterable[str]]):
        """Unregisters a given key from the registry. If the provided keyspace is not in the registry, this method does
        nothing.

        Args:
            keyspace (Iterable[str]): The keyspace (or key if just a str is provided) to retrieve from. This cannot be
                                      empty.
        """
        if isinstance(keyspace, str):
            keyspace = [keyspace]

        self.registry.delete(keyspace)
