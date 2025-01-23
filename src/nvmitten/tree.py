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
from enum import Enum, unique, auto
from typing import Any, Callable, Dict, List, Iterable, Iterator, Optional, Union


@unique
class Traversal(Enum):
    InOrder = auto()
    PreOrder = auto()
    PostOrder = auto()
    OnlyLeaves = auto()


class Tree:
    """Generic Tree implementation. Each node can have any number of children, of any type.
    """

    def __init__(self, name: str, value: Any, children: Optional[Iterable[Tree]] = None):
        """Creates a Tree node.

        Args:
            name (str): ID used to query for this node. Names must be unique among other immediate child nodes of the
                        same parent node.
            value (Any): The value to store for the node.
            children (Iterable[Tree]): If set, initializes this nodes children to this. Otherwise, initialized with no
                                       child nodes. (Default: None)
        """
        self.name = name
        self.value = value
        self.children = dict()
        if children:
            for child in children:
                self.add_child(child)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, value={self.value})"

    def __str__(self) -> str:
        return f"({self.name}: {self.value})"

    def __hash__(self) -> int:
        return hash(self.name) + 17 * hash(self.value)

    def get_children(self) -> List[Tree]:
        """Returns a list of immediate child nodes of the tree.

        Returns:
            List[Tree]: A list of child nodes
        """
        return list(self.children.values())

    def traversal(self, order=Traversal.PreOrder) -> Iterator[Tree]:
        """Returns a generator of *all* nodes in the tree using this node as the root node.

        Args:
            order (Traversal): The order to traverse the tree in. Note that InOrder traversal can only be used if the
                               Tree is a Binary tree, since it is not well-defined in other cases. (Default:
                               Traversal.PreOrder)

        Returns:
            Iterator[Tree]: An Iterator that traverses the tree in the specified order.

        Raises:
            ValueError: If an invalid traversal method is given
        """
        bag = list()
        _get_next = bag.pop
        seen = set()

        def _recurse(node):
            if node in seen:
                raise ValueError(f"Non-tree structure detected - {node} occurs multiple times")
            seen.add(node)

            if order is Traversal.InOrder and (num_children := len(node.children)) > 2:
                raise ValueError(f"{order} is only valid for Binary Trees, but node {node.name} has {num_children} "
                                 "children.")

            if order is Traversal.PostOrder:
                bag.append((node,))  # tuple marks recursion base-case, since a Tree of Trees is possible.

            if order is Traversal.InOrder:
                children = node.get_children()
                if len(children) == 2:  # Right node first
                    bag.append(children[1])
                bag.append((node,))
                if len(children) >= 1:  # Left node
                    bag.append(children[0])
            else:
                # bag.extend(node.children[::-1]) runs through the list twice and is inefficient
                for _, child in reversed(node.children.items()):  # reversed() since bag is a stack
                    bag.append(child)

            if order is Traversal.PreOrder:
                bag.append((node,))

            # Only-Leaf traveral is a pre-order traversal, but only leaves are inserted
            if order is Traversal.OnlyLeaves and len(node.children) == 0:
                bag.append((node,))

        _recurse(self)
        while len(bag) > 0:
            curr = _get_next()
            if isinstance(curr, tuple):
                yield curr[0]
            else:
                _recurse(curr)

    def as_list(self, order=Traversal.PreOrder) -> List[Tree]:
        """Returns a list of *all* nodes in the tree using this node as the root node.

        Args:
            order (Traversal): The order to traverse the tree in. Note that InOrder traversal can only be used if the
                               Tree is a Binary tree, since it is not well-defined in other cases. (Default:
                               Traversal.PreOrder)

        Returns:
            List[Tree]: A list of nodes in the tree in the specified order
        """
        return list(self.traversal(order=order))

    def num_leaves(self) -> int:
        """Returns the number of leaf nodes in this tree.

        Returns:
            int: The number of leaf nodes
        """
        return len(self.as_list(order=Traversal.OnlyLeaves))

    def has_walk(self, keyspace: Iterable[str]) -> bool:
        """Checks if a given walk is valid starting from this node.

        Ex. Given the tree below:
            A
            |- B
            |  |- C
            |  |_ D
            |     |_ E
            |- F
            |  |_ G
            |_ H

        A.has_walk(["B", "D", "E"]) is True, but A.has_walk(["F", "H"]) is False.

        Args:
            keyspace (Iterable[str]): The walk to check

        Returns:
            bool: Whether or not the walk exists from this node
        """
        curr = self
        for key in keyspace:
            if key not in curr.children:
                return False
            curr = curr.children[key]
        return True

    def add_child(self, child_node: Tree):
        """Adds a node as a child to this node. The child_node cannot share a name with an existing child of this node.
        The child node must be a subclass of this node.

        Args:
            child_node (Tree): The node to add as a child

        Raises:
            ValueError: If the name of `child_node` already exists in `self`.
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
        """Inserts a value into this node. The created child node will be the same class as its immediate parent.

        Args:
            name (str): The name of the node to be created
            value (Any): The value to insert
            keyspace (Iterable[str]): If specified, instead inserts the value in the node at the end of the walk
                                      starting from `self` represented by the names of the nodes given in
                                      `keyspace`. (Default: None)
            create_parents (bool): If True, will create any necessary parent nodes for the walk for the given keyspace
                                   using the `default_parent_value`.
            default_parent_value (Any): Default value for parent nodes.

        Returns:
            Tree: The Tree that was created

        Raises:
            KeyError: If the walk specified by `keyspace` is an invalid walk of the Tree
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
        """Gets a value of the node in the tree given a keyspace that represents the names of nodes in a walk of the
        tree.

        If the walk is invalid, raises an error.

        Args:
            keyspace (Iterable[str]): A list of strings representing names of the nodes in a walk of the tree.

        Raises:
            KeyError: If `keyspace` is an invalid walk
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
            keyspace (Iterable[str]): A list of strings representing names of the nodes in a walk of the tree. This
                                      cannot be empty.
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
            keyspace (Iterable[str]): A list of strings representing names of the nodes in a walk of the tree. This
                                      cannot be empty.
        """
        if len(keyspace) == 0:
            raise ValueError(".delete cannot be called with an empty keyspace. Use `del` instead.")

        curr = self
        for name in keyspace:
            if name not in curr.children:
                return False
            curr = curr.children[name]
        return True

    def __getitem__(self, keyspace: Union[str, Iterable[str]]):
        """See `get()`.

        `node[('child', 'grandchild')]` will return the node named `grandchild` that is a depth of 2 from `self`.
        """
        if isinstance(keyspace, str):
            keyspace = [keyspace]
        return self.get(keyspace)

    def __setitem__(self, keyspace: Union[str, Iterable[str]], value):
        """See `insert_value()`.

        The last value of `keyspace` is assumed to be the name of the node to create.

        `node[('child', 'grandchild')] = 5` will insert a node named 'grandchild' with a value of 5 in the node named
        'child' that is a depth of 1 from `self`.
        """
        if isinstance(keyspace, str):
            keyspace = [keyspace]
        self.insert_value(keyspace[-1], value, keyspace=keyspace[:-1])
