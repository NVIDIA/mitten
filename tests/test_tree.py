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


import pytest

from nvmitten.tree import Tree, Traversal


class TestTree:
    """Tests the methods for Tree
    """

    def test_get_children_added(self):
        t1 = Tree("root", None)
        t2 = Tree("child1", 1)
        t3 = Tree("child2", 2)

        # Manual setting of children
        t1.children["child1"] = t2
        t1.children["child2"] = t3

        children = t1.get_children()

        assert isinstance(children, list)
        assert len(children) == 2
        assert children[0] is t2  # Children should store refs, not values
        assert children[1] is t3

    def test_get_children_constructor(self):
        child1 = Tree("child1", 1)
        child2 = Tree("child2", 2)
        child3 = Tree("child3", 3)

        root = Tree("root", None, children=[child1, child2, child3])
        children = root.get_children()

        assert isinstance(children, list)
        assert len(children) == 3
        assert children[0] is child1
        assert children[1] is child2
        assert children[2] is child3

    def test_traversal_preorder(self):
        """Test structure:

        A
        |- B
        |  |- C
        |  |_ D
        |     |_ E
        |- F
        |  |_ G
        |_ H
        """
        H = Tree("H", 1)
        G = Tree("G", 2)
        F = Tree("F", 3, children=[G])
        E = Tree("E", 4)
        D = Tree("D", 5, children=[E])
        C = Tree("C", 6)
        B = Tree("B", 7, children=[C, D])
        A = Tree("A", 8, children=[B, F, H])

        traversal = list(A.traversal(order=Traversal.PreOrder))
        assert traversal == [A, B, C, D, E, F, G, H]
        assert traversal == A.as_list(order=Traversal.PreOrder)

    def test_traversal_postorder(self):
        """Test structure:

        A
        |- B
        |  |- C
        |  |_ D
        |     |_ E
        |- F
        |  |_ G
        |_ H
        """
        H = Tree("H", 1)
        G = Tree("G", 2)
        F = Tree("F", 3, children=[G])
        E = Tree("E", 4)
        D = Tree("D", 5, children=[E])
        C = Tree("C", 6)
        B = Tree("B", 7, children=[C, D])
        A = Tree("A", 8, children=[B, F, H])

        traversal = list(A.traversal(order=Traversal.PostOrder))
        assert traversal == [C, E, D, B, G, F, H, A]
        assert traversal == A.as_list(order=Traversal.PostOrder)

    def test_traversal_inorder(self):
        """Test structure:

        A
        |- B
        |  |- C
        |  |_ D
        |     |_ E
        |_ F
           |_ G
        """
        G = Tree("G", 2)
        F = Tree("F", 3, children=[G])
        E = Tree("E", 4)
        D = Tree("D", 5, children=[E])
        C = Tree("C", 6)
        B = Tree("B", 7, children=[C, D])
        A = Tree("A", 8, children=[B, F])

        traversal = list(A.traversal(order=Traversal.InOrder))
        assert traversal == [C, B, E, D, A, G, F]
        assert traversal == A.as_list(order=Traversal.InOrder)

    def test_traversal_inorder_nonbinary(self):
        """Test structure:

        A
        |- B
        |  |- C
        |  |_ D
        |     |_ E
        |- F
        |  |_ G
        |_ H
        """
        H = Tree("H", 1)
        G = Tree("G", 2)
        F = Tree("F", 3, children=[G])
        E = Tree("E", 4)
        D = Tree("D", 5, children=[E])
        C = Tree("C", 6)
        B = Tree("B", 7, children=[C, D])
        A = Tree("A", 8, children=[B, F, H])

        with pytest.raises(ValueError) as err:
            traversal = list(A.traversal(order=Traversal.InOrder))

        err_msg = err.value.args[0]
        assert err_msg == f"{Traversal.InOrder} is only valid for Binary Trees, but node A has 3 children."

    def test_traversal_onlyleaves(self):
        """Test structure:

        A
        |- B
        |  |- C
        |  |_ D
        |     |_ E
        |- F
        |  |_ G
        |_ H
        """
        H = Tree("H", 1)
        G = Tree("G", 2)
        F = Tree("F", 3, children=[G])
        E = Tree("E", 4)
        D = Tree("D", 5, children=[E])
        C = Tree("C", 6)
        B = Tree("B", 7, children=[C, D])
        A = Tree("A", 8, children=[B, F, H])

        traversal = list(A.traversal(order=Traversal.OnlyLeaves))
        assert traversal == [C, E, G, H]
        assert traversal == A.as_list(order=Traversal.OnlyLeaves)

    def test_num_leaves(self):
        """Test structure:

        A
        |- B
        |  |- C
        |  |_ D
        |     |_ E
        |- F
        |  |_ G
        |_ H
        """
        H = Tree("H", 1)
        G = Tree("G", 2)
        F = Tree("F", 3, children=[G])
        E = Tree("E", 4)
        D = Tree("D", 5, children=[E])
        C = Tree("C", 6)
        B = Tree("B", 7, children=[C, D])
        A = Tree("A", 8, children=[B, F, H])

        assert A.num_leaves() == 4
        assert B.num_leaves() == 2
        assert C.num_leaves() == 1
        assert D.num_leaves() == 1
        assert E.num_leaves() == 1
        assert F.num_leaves() == 1
        assert G.num_leaves() == 1
        assert H.num_leaves() == 1

    def test_add_child(self):
        """Test structure:

        A
        |- B
        |  |- C
        |  |_ D
        |     |_ E
        |- F
        |  |_ G
        |_ H
        """
        H = Tree("H", 1)
        G = Tree("G", 2)
        F = Tree("F", 3)
        E = Tree("E", 4)
        D = Tree("D", 5)
        C = Tree("C", 6)
        B = Tree("B", 7)
        A = Tree("A", 8)

        F.add_child(G)
        D.add_child(E)
        B.add_child(C)
        B.add_child(D)
        A.add_child(B)
        A.add_child(F)
        A.add_child(H)

        traversal = list(A.traversal(order=Traversal.PreOrder))
        assert traversal == [A, B, C, D, E, F, G, H]
        assert traversal == A.as_list(order=Traversal.PreOrder)

    def test_add_child_existing(self):
        A = Tree("A", 1)
        B = Tree("B", 2)
        A.add_child(B)
        assert len(A.get_children()) == 1

        with pytest.raises(ValueError) as err:
            A.add_child(B)

        err_msg = err.value.args[0]
        assert err_msg == f"Cannot insert node with duplicate name B"

    def test_add_child_invalidtype(self):
        class MyTree(Tree):
            pass

        A = Tree("A", 1)
        B = MyTree("B", 2)
        A.add_child(B)  # Should not crash
        assert len(A.get_children()) == 1

        A = MyTree("A", 1)
        B = Tree("B", 2)

        with pytest.raises(AssertionError) as err:
            A.add_child(B)

        err_msg = err.value.args[0]
        assert err_msg == f"Cannot add child node Tree. Must be subclass of MyTree"

    def test_delete(self):
        """Test structure:

        A
        |- B
        |  |- C
        |  |_ D
        |     |_ E
        |- F
        |  |_ G
        |_ H
        """
        H = Tree("H", 1)
        G = Tree("G", 2)
        F = Tree("F", 3, children=[G])
        E = Tree("E", 4)
        D = Tree("D", 5, children=[E])
        C = Tree("C", 6)
        B = Tree("B", 7, children=[C, D])
        A = Tree("A", 8, children=[B, F, H])

        traversal = list(A.traversal(order=Traversal.PreOrder))
        assert traversal == [A, B, C, D, E, F, G, H]
        assert traversal == A.as_list(order=Traversal.PreOrder)
        assert "D" in A["B"].children
        A.delete(["B", "D"])
        assert "D" not in A["B"].children
        assert E is not None  # We just need to make sure E exists and is not GC'd up

    def test_cyclic_detect(self):
        A = Tree("A", 1)
        B = Tree("B", 2)
        A.add_child(B)
        B.add_child(A)

        with pytest.raises(ValueError) as err:
            A.as_list()

        err_msg = err.value.args[0]
        assert err_msg == f"Non-tree structure detected - {A} occurs multiple times"

    def test_insert_value(self):
        """Test structure:

        A
        |- B
        |  |- C
        |  |_ D
        |     |_ E
        |- F
        |  |_ G
        |_ H
        """
        A = Tree("A", 1)
        B = A.insert_value("B", 2)
        C = B.insert_value("C", 3)
        D = B.insert_value("D", 4)
        E = D.insert_value("E", 5)
        F = A.insert_value("F", 6)
        G = F.insert_value("G", 7)
        H = A.insert_value("H", 8)

        assert [A, B, C, D, E, F, G, H] == A.as_list(order=Traversal.PreOrder)
        assert [C, E, D, B, G, F, H, A] == A.as_list(order=Traversal.PostOrder)

    def test_insert_value_keyspace(self):
        """Test structure:

        A
        |- B
        |  |- C
        |  |_ D
        |     |_ E
        |- F
        |  |_ G
        |_ H
        """
        A = Tree("A", 1)
        B = A.insert_value("B", 2)
        C = A.insert_value("C", 3, keyspace=("B",))
        D = A.insert_value("D", 4, keyspace=("B",))
        E = A.insert_value("E", 5, keyspace=("B", "D"))
        F = A.insert_value("F", 6)
        G = A.insert_value("G", 7, keyspace=("F",))
        H = A.insert_value("H", 8)

        assert [A, B, C, D, E, F, G, H] == A.as_list(order=Traversal.PreOrder)
        assert [C, E, D, B, G, F, H, A] == A.as_list(order=Traversal.PostOrder)

    def test_insert_value_dunder(self):
        """Test structure:

        A
        |- B
        |  |- C
        |  |_ D
        |     |_ E
        |- F
        |  |_ G
        |_ H
        """
        A = Tree("A", 1)
        A["B"] = 2
        A[("B", "C")] = 3
        A[("B", "D")] = 4
        A[("B", "D", "E")] = 5
        A["F"] = 6
        A["F", "G"] = 7
        A["H"] = 8

        B = A["B"]
        C = A["B", "C"]
        D = A["B", "D"]
        E = A["B", "D", "E"]
        F = A["F"]
        G = A["F", "G"]
        H = A["H"]

        assert [A, B, C, D, E, F, G, H] == A.as_list(order=Traversal.PreOrder)
        assert [C, E, D, B, G, F, H, A] == A.as_list(order=Traversal.PostOrder)
