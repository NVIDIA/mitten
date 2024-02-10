# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


@pytest.mark.parametrize(
    "ssroot,namespace",
    [
        ("/tmp/nvmitten/test/scratch_space", "a1"),
        ("/tmp/nvmitten/test/scratch_space", "a2"),
        ("/tmp/nvmitten/test/scratch_space", "a3"),
        ("/tmp/nvmitten/test/scratch_space", ""),
    ]
)
def test_scratch_space(ssroot, namespace):
    # This test directly modifies the filesystem, so there is a small chance the test sporadically reports a false
    # positive due to a directory existing due to external reasons, or a false negative if the directory was deleted by
    # external reasons.
    p = Path(ssroot)
    if p.exists():
        shutil.rmtree(p)
    assert not p.exists()

    ss = ScratchSpace(ssroot)
    assert not p.exists()
    ss.create()
    assert p.exists()
    wd = ss.working_dir(namespace=namespace)
    assert wd.exists()


@pytest.mark.parametrize(
    "b1,b2,expected",
    [
        (BenchmarkMetric("img/s"), BenchmarkMetric("images per second"), False),
        (BenchmarkMetric("img/s"), BenchmarkMetric("img/s"), True),
        (BenchmarkMetric("img/s", bigger_is_better=False), BenchmarkMetric("images per second"), False),
        (BenchmarkMetric("img/s"), BenchmarkMetric("img/s", bigger_is_better=False), False)
    ]
)
def test_benchmark_metric(b1, b2, expected):
    # TODO: BenchmarkMetric usage isn't well defined yet. This test isn't made to be too robust, but is just for
    # coverage.
    assert (b1 == b2) is expected


class O1(Operation):
    def __init__(self, retval=True):
        self.retval = retval

    @classmethod
    def immediate_dependencies(cls):
        return None

    def run(self, scratch_space, dependency_outputs):
        assert len(dependency_outputs) == 0  # O1 has no dependencies, so should receive no input.
        return self.retval


class O2(Operation):
    @classmethod
    def immediate_dependencies(cls):
        return [O1]

    def run(self, scratch_space, dependency_outputs):
        assert len(dependency_outputs) == 0  # O1 is a dependency, but has no output values
        return {"out2.1": "hello",
                "out2.2": "world"}


class O3(Operation):
    @classmethod
    def immediate_dependencies(cls):
        return [O1, O2]

    def run(self, scratch_space, dependency_outputs):
        # Test 02 outputs are forwarded correctly in dependency_outputs
        assert len(dependency_outputs) == 1
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
        assert O3 in dependency_outputs
        assert dependency_outputs[O2] == {"out2.1": "hello", "out2.2": "world"}
        assert dependency_outputs[O3] == {"out3.1": "foo"}
        return {"out4.1": "bar"}


def test_pipeline_toposort():
    p = Pipeline(None, [O2, O3, O1, O4], dict())
    assert p.topo_sort() == ((O1, O2, O3, O4), dict())

    p = Pipeline(None, [O1], dict())
    assert p.topo_sort() == ((O1,), dict())


def test_pipeline_config():
    p = Pipeline(None, [O1], {O1: {"retval": {"out1.1": 1234}}})
    assert p.run() == {"out1.1": 1234}


def test_pipeline_mark_output():
    p = Pipeline(None, [O2, O3, O1, O4], dict())
    p.mark_output(O3)
    assert p.run() == {"out3.1": "foo"}


class FooImpl(Impl):
    @classmethod
    def outputs(cls):
        return ["foo1", "foo2"]


class MyFoo(Operation):
    @classmethod
    def implements(cls):
        return FooImpl

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
        return [FooImpl]

    def run(self, scratch_space, dependency_outputs):
        assert len(dependency_outputs) == 1
        assert FooImpl in dependency_outputs  # Key is Impl not Operation
        assert "discarded" not in dependency_outputs[FooImpl]
        assert dependency_outputs[FooImpl] == {"foo1": "helloworld", "foo2": "worldhello"}
        return True


def test_pipeline_impl():
    p = Pipeline(None, [O1, O2, MyFoo, Bar], dict())
    ordering, impls = p.topo_sort()
    assert ordering == (O1, O2, MyFoo, Bar)
    assert impls == {FooImpl: MyFoo}
    assert p.run()


def test_pipeline_impl_missing():
    with pytest.raises(ImplementationNotFoundError):
        p = Pipeline(None, [O2, Bar], dict())
        p.run()
        assert False  # Should not be reached


def test_too_many_impls():
    class Foofoo(Operation):
        @classmethod
        def implements(cls):
            return FooImpl

        @classmethod
        def immediate_dependencies(cls):
            return None

        def run(self, scratch_space, dependency_outputs):
            return True

    with pytest.raises(TooManyImplementationsError):
        p = Pipeline(None, [O2, MyFoo, Foofoo, Bar], dict())
        p.run()
        assert False  # Should not be reached


def test_invalid_impl():
    class BadImplOp(Operation):
        @classmethod
        def implements(cls):
            return O1  # Implements an Op instead of Impl

        @classmethod
        def immediate_dependencies(cls):
            return None

        def run(self, scratch_space, dependency_outputs):
            return True

    with pytest.raises(InvalidImplError):
        Pipeline(None, [BadImplOp], dict()).run()
        assert False
