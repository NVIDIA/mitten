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

"""Tree data structure implementation for hierarchical data organization.

This module provides a generic tree implementation that supports:
- Arbitrary number of children per node
- Multiple traversal methods (InOrder, PreOrder, PostOrder, OnlyLeaves)
- Dictionary-like access to nodes
- Tree construction from dictionaries
- Node insertion and deletion
- Path-based node access
"""

from __future__ import annotations
from enum import Enum, unique, auto
from typing import Any, Callable, Dict, List, Iterable, Iterator, Optional, Union

import math


@unique
class Traversal(Enum):
    """Enumeration of tree traversal methods.

    Attributes:
        InOrder: Visit left subtree, then root, then right subtree (valid only for binary trees)
        PreOrder: Visit root, then children from left to right
        PostOrder: Visit children from left to right, then root
        OnlyLeaves: Visit only leaf nodes in pre-order
    """
    InOrder = auto()
    PreOrder = auto()
    PostOrder = auto()
    OnlyLeaves = auto()


class Tree:
    """Generic Tree implementation for hierarchical data organization.

    This class provides a flexible tree structure where:
    - Each node can have any number of children
    - Children are stored in a dictionary keyed by their names
    - Nodes can store arbitrary values
    - Supports multiple traversal methods
    - Provides dictionary-like access to nodes
    """

    def __init__(self, name: str, value: Any, children: Optional[Iterable[Tree]] = None):
        """Creates a Tree node.

        Args:
            name: Unique identifier for this node among its siblings
            value: Arbitrary value to store in this node
            children: Optional initial set of child nodes

        Raises:
            ValueError: If a child node has a duplicate name
        """
        self.name = name
        self.value = value
        self.children = dict()
        if children:
            for child in children:
                self.add_child(child)

    @classmethod
    def from_dict(cls, d: Dict, name: str, max_depth: int = math.inf) -> Tree:
        """Creates a tree from a nested dictionary structure.

        Args:
            d: Dictionary to convert to a tree
            name: Name for the root node
            max_depth: Maximum depth to convert (default: infinite)

        Returns:
            Tree: Root node of the constructed tree

        Raises:
            AssertionError: If max_depth is less than 1
        """
        root = cls(name, None)

        assert max_depth >= 1, "Max nested depth must be at least 1"

        for k, v in d.items():
            if isinstance(v, dict) and max_depth > 1:
                _subdepth = max_depth - 1
                root.add_child(cls.from_dict(v, k, max_depth=_subdepth))
            else:
                root[k] = v

        return root

    def __repr__(self) -> str:
        """Returns a string representation of the node.

        Returns:
            str: String in format "Tree(name=name, value=value)"
        """
        return f"{self.__class__.__name__}(name={self.name}, value={self.value})"

    def __str__(self) -> str:
        """Returns a simplified string representation of the node.

        Returns:
            str: String in format "(name: value)"
        """
        return f"({self.name}: {self.value})"

    def __hash__(self) -> int:
        """Returns a hash of the node based on its id.

        This is used to track visited nodes during traversal.

        Returns:
            int: Hash value based on node's id
        """
        return id(self)

    def get_children(self) -> List[Tree]:
        """Returns a list of immediate child nodes.

        Returns:
            List[Tree]: List of child nodes
        """
        return list(self.children.values())

    def traversal(self, order: Traversal = Traversal.PreOrder, include_keys: bool = False) -> Iterator[Tree]:
        """Returns an iterator over all nodes in the tree.

        Args:
            order: Traversal order (default: PreOrder)
            include_keys: Whether to include node paths in the output

        Returns:
            Iterator[Tree]: Iterator over nodes in specified order

        Raises:
            ValueError: If InOrder traversal is used on a non-binary tree
            ValueError: If a cycle is detected in the tree
        """
        bag = list()
        _get_next = bag.pop
        seen = set()

        def _add(stop, node, keyspace):
            if include_keys:
                bag.append((stop, node, keyspace))
            else:
                bag.append((stop, node))

        def _recurse(node, keyspace):
            if node in seen:
                raise ValueError(f"Non-tree structure detected - {node} occurs multiple times")
            seen.add(node)

            if order is Traversal.InOrder and (num_children := len(node.children)) > 2:
                raise ValueError(f"{order} is only valid for Binary Trees, but node {node.name} has {num_children} "
                                 "children.")

            if order is Traversal.PostOrder:
                _add(True, node, keyspace)

            if order is Traversal.InOrder:
                children = node.get_children()
                if len(children) == 2:  # Right node first
                    _add(False, children[1], [*keyspace, children[1].name])

                _add(True, node, keyspace)

                if len(children) >= 1:  # Left node
                    _add(False, children[0], [*keyspace, children[0].name])
            else:
                for _, child in reversed(node.children.items()):
                    _add(False, child, [*keyspace, child.name])

            if order is Traversal.PreOrder:
                _add(True, node, keyspace)

            if order is Traversal.OnlyLeaves and len(node.children) == 0:
                _add(True, node, keyspace)

        _recurse(self, list())
        while len(bag) > 0:
            if include_keys:
                stop, curr, keyspace = _get_next()
            else:
                stop, curr = _get_next()

            if stop:
                if include_keys:
                    yield curr, keyspace
                else:
                    yield curr
            else:
                _recurse(curr, keyspace if include_keys else list())

    def as_list(self, order: Traversal = Traversal.PreOrder) -> List[Tree]:
        """Returns a list of all nodes in the tree.

        Args:
            order: Traversal order (default: PreOrder)

        Returns:
            List[Tree]: List of nodes in specified order
        """
        return list(self.traversal(order=order))

    def num_leaves(self) -> int:
        """Returns the number of leaf nodes in the tree.

        Returns:
            int: Number of leaf nodes
        """
        return len(self.as_list(order=Traversal.OnlyLeaves))

    def has_walk(self, keyspace: Iterable[str]) -> bool:
        """Checks if a path exists in the tree.

        Example:
            Given tree:
                A
                |- B
                |  |- C
                |  |_ D
                |     |_ E
                |- F
                |  |_ G
                |_ H

            A.has_walk(["B", "D", "E"]) returns True
            A.has_walk(["F", "H"]) returns False

        Args:
            keyspace: Sequence of node names representing the path

        Returns:
            bool: True if the path exists, False otherwise
        """
        curr = self
        for key in keyspace:
            if key not in curr.children:
                return False
            curr = curr.children[key]
        return True

    def add_child(self, child_node: Tree):
        """Adds a child node to this node.

        Args:
            child_node: Node to add as a child

        Raises:
            AssertionError: If child_node is not a subclass of this node's class
            ValueError: If a child with the same name already exists
        """
        assert isinstance(child_node, self.__class__), f"Cannot add child node {child_node.__class__.__name__}. "\
                                                       f"Must be subclass of {self.__class__.__name__}"
        if child_node.name in self.children:
            raise ValueError(f"Cannot insert node with duplicate name {child_node.name}")
        self.children[child_node.name] = child_node

    def insert_value(self,
                     name: str,
                     value: Any,
                     keyspace: Optional[Iterable[str]] = None,
                     create_parents: bool = True,
                     default_parent_value: Any = None) -> Tree:
        """Inserts a value into the tree at the specified location.

        Args:
            name: Name of the new node
            value: Value to store in the new node
            keyspace: Path to the parent node (default: None for root)
            create_parents: Whether to create missing parent nodes
            default_parent_value: Value to use for created parent nodes

        Returns:
            Tree: The newly created node

        Raises:
            KeyError: If keyspace is invalid and create_parents is False
        """
        curr = self
        if keyspace:
            for key in keyspace:
                if create_parents and key not in curr.children:
                    curr.children[key] = curr.__class__(key, default_parent_value)
                curr = curr.children[key]
        new_node = curr.__class__(name, value)
        curr.add_child(new_node)
        return new_node

    def get(self, keyspace: Iterable[str]) -> Any:
        """Retrieves a value from the tree.

        Args:
            keyspace: Path to the target node

        Returns:
            Any: Value stored in the target node

        Raises:
            KeyError: If the path does not exist
        """
        curr = self
        for name in keyspace:
            curr = curr.children[name]
        return curr

    def delete(self, keyspace: Iterable[str]):
        """Deleted a node in a tree (if exists). If the node does not exist, this method does nothing.

        WARNING: Be careful when calling this method! If a non-leaf node is deleted and there are no references to any
        of the node's children, they may be lost to the whims Python GC. This method does NOT delete children of the
        node being deleted.

        Args:
            keyspace: Path to the node to delete

        Raises:
            KeyError: If the path does not exist
        """
        if len(keyspace) == 0:
            raise ValueError(".delete cannot be called with an empty keyspace. Use `del` instead.")

        curr = self
        for name in keyspace[:-1]:
            if name not in curr.children:
                return
            curr = curr.children[name]
        if keyspace[-1] not in curr.children:
            return
        else:
            del curr.children[keyspace[-1]]

    def __contains__(self, keyspace: Iterable[str]):
        """Checks if a node exists in this Tree.

        Args:
            keyspace: Path to check

        Returns:
            bool: True if the path exists, False otherwise
        """
        if len(keyspace) == 0:
            raise ValueError("Tree.contains cannot be called with an empty keyspace.")

        curr = self
        for name in keyspace:
            if name not in curr.children:
                return False
            curr = curr.children[name]
        return True

    def __getitem__(self, keyspace: Union[str, Iterable[str]]) -> Any:
        """Dictionary-like access to node values.

        Args:
            keyspace: Single key or path to the target node

        Returns:
            Any: Value stored in the target node

        Raises:
            KeyError: If the path does not exist
        """
        if isinstance(keyspace, str):
            keyspace = [keyspace]
        return self.get(keyspace)

    def __setitem__(self, keyspace: Union[str, Iterable[str]], value: Any):
        """Dictionary-like assignment to node values.

        Args:
            keyspace: Single key or path to the target node
            value: Value to store

        Raises:
            KeyError: If the path does not exist
        """
        if not isinstance(keyspace, list) and not isinstance(keyspace, tuple):
            keyspace = [keyspace]
        self.insert_value(keyspace[-1], value, keyspace=keyspace[:-1])
