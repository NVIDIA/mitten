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

from pathlib import Path

import pytest
import shutil

from nvmitten.pipeline import *


class O1(Operation):
    def __init__(self, retval=True):
        self.retval = retval

    @classmethod
    def output_keys(cls):
        return ["out1.1"]

    @classmethod
    def immediate_dependencies(cls):
        return None

    def run(self, scratch_space, dependency_outputs):
        assert len(dependency_outputs) == 0  # O1 has no dependencies, so should receive no input.
        return {"out1.1": self.retval}


class O2(Operation):
    @classmethod
    def output_keys(cls):
        return ["out2.1", "out2.2"]

    @classmethod
    def immediate_dependencies(cls):
        return [O1]

    def run(self, scratch_space, dependency_outputs):
        assert len(dependency_outputs) == 1  # O1 is a dependency
        assert O1 in dependency_outputs
        return {"out2.1": "hello",
                "out2.2": "world"}


class O3(Operation):
    @classmethod
    def output_keys(cls):
        return ["out3.1"]

    @classmethod
    def immediate_dependencies(cls):
        return [O1, O2]

    def run(self, scratch_space, dependency_outputs):
        # Test 02 outputs are forwarded correctly in dependency_outputs
        assert len(dependency_outputs) == 2
        assert O1 in dependency_outputs
        assert O2 in dependency_outputs
        assert dependency_outputs[O2] == {"out2.1": "hello", "out2.2": "world"}
        return {"out3.1": "foo"}


class O4(Operation):
    @classmethod
    def immediate_dependencies(cls):
        return [O2, O3]

    def run(self, scratch_space, dependency_outputs):
        # Test 02 and 03 outputs were both forwarded correctly
        assert len(dependency_outputs) == 2
        assert O2 in dependency_outputs
        assert dependency_outputs[O2] == {"out2.1": "hello", "out2.2": "world"}

        assert O3 in dependency_outputs
        assert dependency_outputs[O3] == {"out3.1": "foo"}
        return {"out4.1": "bar"}


def test_pipeline_toposort(debug_manager_io_stream):
    p = Pipeline(None, [O2, O3, O1, O4], dict())
    assert p.topo_sort() == ((O1, O2, O3, O4), dict())

    p = Pipeline(None, [O1], dict())
    assert p.topo_sort() == ((O1,), dict())


def test_pipeline_config(debug_manager_io_stream):
    p = Pipeline(None, [O1], {O1: {"retval": 1234}})
    assert p.run() == {"out1.1": 1234}


def test_pipeline_mark_output(debug_manager_io_stream):
    p = Pipeline(None, [O2, O3, O1, O4], dict())
    p.mark_output(O3)
    assert p.run() == {"out3.1": "foo"}


def test_pipeline_early_stop(debug_manager_io_stream):
    p = Pipeline(None, [O2, O3, O1, O4], dict())
    p.mark_output(O3)
    result_noES = p.run()
    assert result_noES == {"out3.1": "foo"}
    assert p._cache is not None
    assert p._cache[O4].status is OperationStatus.PASSED
    assert p._cache[O4].value == {"out4.1": "bar"}

    result_ES = p.run(early_stop=True)
    assert result_ES == {"out3.1": "foo"}
    assert p._cache is not None
    assert p._cache[O4].status is OperationStatus.SKIPPED
    assert p._cache[O4].value is None


class AbstractFoo(Operation):
    @classmethod
    def output_keys(cls):
        return ["foo1", "foo2"]


class MyFoo(AbstractFoo):
    @classmethod
    def output_keys(cls):
        return ["foo1", "foo2"]

    @classmethod
    def immediate_dependencies(cls):
        return [O2]

    def run(self, scratch_space, dependency_outputs):
        s1 = dependency_outputs[O2]["out2.1"]
        s2 = dependency_outputs[O2]["out2.2"]
        return {"foo1": s1 + s2,
                "foo2": s2 + s1,
                "discarded": 12345}


class Bar(Operation):
    @classmethod
    def immediate_dependencies(cls):
        return [AbstractFoo]

    def run(self, scratch_space, dependency_outputs):
        assert len(dependency_outputs) == 1
        assert AbstractFoo in dependency_outputs  # Key is Abstract, not implementation
        assert "discarded" not in dependency_outputs[AbstractFoo]
        assert dependency_outputs[AbstractFoo] == {"foo1": "helloworld", "foo2": "worldhello"}
        # No output keys


def test_pipeline_abstracts(debug_manager_io_stream):
    p = Pipeline(None, [O1, O2, MyFoo, Bar], dict())
    ordering, impls = p.topo_sort()
    assert ordering == (O1, O2, MyFoo, Bar)
    assert impls == {AbstractFoo: MyFoo}
    p.run()


def test_pipeline_impl_missing(debug_manager_io_stream):
    with pytest.raises(ImplementationNotFoundError):
        p = Pipeline(None, [O2, Bar], dict())
        p.run()
        assert False, "Exception should have already been raised"  # Should not be reached


def test_multiple_implementations(debug_manager_io_stream):
    class ExtraFoo(AbstractFoo):
        @classmethod
        def output_keys(cls):
            return ["foo1", "foo2"]

        @classmethod
        def immediate_dependencies(cls):
            return None

        def run(self, scratch_space, dependency_outputs):
            return {"foo1": 123, "foo2": 456}

    p = Pipeline(None, [O1, O2, MyFoo, ExtraFoo, Bar], dict())
    ordering, impls = p.topo_sort()
    assert len(impls) == 1
    assert impls[AbstractFoo] is MyFoo
    assert ExtraFoo in ordering  # ExtraFoo should still be in the graph to be executed.
    # If run() mistakenly gives ExtraFoo's output to Bar, Bar.run will throw an AssertionError
    p.run()


def test_invalid_implementation(debug_manager_io_stream):
    with pytest.raises(MissingParentOutputKey):
        # Exception should be raised upon class declaration.
        class BadImplementation(O1):
            @classmethod
            def output_keys(cls):
                return ["bad"]

            @classmethod
            def immediate_dependencies(cls):
                return None

            def run(self, scratch_space, dependency_outputs):
                return {"bad": True}

    with pytest.raises(MissingParentOutputKey):
        # Exception should be raised upon class declaration.
        class BadImplementation(AbstractFoo):
            @classmethod
            def output_keys(cls):
                return ["foo1"]

            @classmethod
            def immediate_dependencies(cls):
                return None

            def run(self, scratch_space, dependency_outputs):
                return {"foo1": True}
